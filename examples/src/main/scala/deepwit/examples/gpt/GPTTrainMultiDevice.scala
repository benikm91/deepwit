package deepwit.examples.gpt
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.CategoricalCrossEntropy

import dimwit.*
import dimwit.Conversions.given
import deepwit.training.{Monitor, after, tapEvery}
import deepwit.optimizer.*
import dimwit.optimizer.{AdamW, Adam, AdamState}
import dimwit.TreeOf.ops.*

import FineWebDataset.batchStream

import dimwit.TreeOf.map

import deepwit.optimizer.LearningRateSchedule
import deepwit.optimizer.LearningRateSchedules.LinearWarmup

import Config.*

/** The mesh axis the batch is split over. */
trait Data derives MeshLabel

/** As FineWebDataset.BatchSample, but with the sample axis split across the mesh. */
case class BatchSample(
    targets: Tensor2[Sample |@| Data, Context, Int32],
    inputs: Tensor2[Sample |@| Data, Context, Int32]
)

/** Data-parallel GPT-2 training across every device the backend reports.
  *
  * Every device runs the running batch that the single-device script ran on one, so the running
  * batch grows with the device count and the accumulation steps needed to reach the same effective
  * batch fall by the same factor. `effectiveBatchSize` is untouched, so each gradient is still
  * taken over 512 samples and the training dynamics are unchanged.
  *
  * The model never sees the batch axis and is untouched. Only `Sample` becomes `Sample |@| Data`.
  *
  * {{{
  * FINEWEB_DIR=/path/to/fineweb10B sbt "examples/runMain deepwit.examples.gpt.trainMultiDevice"
  * }}}
  */
@main def trainMultiDevice(): Unit =
  val devices = dimwit.jax.Jax.devices.size

  // The three Config values that depend on how many devices there are. Each device runs
  // Config.runningBatchSize, so the same effective batch needs proportionally fewer accumulations.
  val runningBatchSize = Config.runningBatchSize * devices
  val accumulationSteps = effectiveBatchSize / runningBatchSize
  val numBatchesPerValidation = valSamples / runningBatchSize

  require(
    effectiveBatchSize % runningBatchSize == 0,
    s"effective batch $effectiveBatchSize is not divisible by the running batch $runningBatchSize ($devices devices)"
  )

  val mesh = Mesh(MeshAxis[Data] -> devices)
  println(s"Training over $mesh: ${mesh.devices.map(d => s"${d.platform}:${d.id}").mkString(", ")}")
  println(s"running batch $runningBatchSize (${Config.runningBatchSize} per device), $accumulationSteps accumulation steps")

  def shardBatch(batch: FineWebDataset.BatchSample): BatchSample =
    BatchSample(
      batch.targets.shard(mesh, Axis[Sample] -> MeshAxis[Data]),
      batch.inputs.shard(mesh, Axis[Sample] -> MeshAxis[Data])
    )

  val key = Random.Key.fromTime()

  val (dataKey, initParamsKey) = key.split2()

  val initParams = GPT.Params.gpt2Init(numTransformerLayers = numLayers)(
    vocabExtent,
    contextExtent,
    numHeads,
    embeddingExtent,
    embeddingMixedExtent,
    initParamsKey,
    VType[Float32]
  )

  val schedule = LinearWarmup(baseLearningRate, 1_000).followBy(CosineDecay(baseLearningRate, minLearningRate, 20_000))
  val opt = LearningRateScheduler(lr => AdamW(Adam(lr, beta1 = beta1, beta2 = beta2), weightDecayFactor = weightDecayFactor), schedule)

  case class TrainingState(
      params: GPT.Params[Float32],
      optState: LearningRateSchedulerState[GPT.Params[Float32], AdamState],
      stepCost: Tensor0[Float32]
  )

  val (trainKey, valKey) = dataKey.split2()
  val dataDir = FineWebDataset.defaultDataDir
  val trainStream = batchStream(dataDir, "fineweb_train_", runningBatchSize, contextExtent.size, trainKey).map(shardBatch)
  val valStream = batchStream(dataDir, "fineweb_val_", runningBatchSize, contextExtent.size, valKey).map(shardBatch)

  def loss[V: IsFloating](
      targets: Tensor1[Context, Int32],
      logits: Tensor2[Context, Vocab, V]
  ): Tensor0[V] =
    zipvmap(Axis[Context])(targets, logits)(CategoricalCrossEntropy.fromLogits).mean

  def costFunFor[V: IsFloating](
      batchSample: BatchSample
  )(
      params: GPT.Params[V]
  ): Tensor0[V] =
    val model = GPT(params)
    val losses = zipvmap(Axis[Sample |@| Data])(batchSample.targets, batchSample.inputs):
      case (targets, inputs) =>
        val logits = model.logits(inputs)
        loss(targets, logits)
    losses.mean
  val jitCostFn = jit: (params: GPT.Params[BFloat16], batchSample: BatchSample) =>
    costFunFor(batchSample)(params)

  def calcGradients(
      batchSample: BatchSample,
      params: GPT.Params[BFloat16]
  ): (Tensor0[BFloat16], Grad[GPT.Params[BFloat16]]) =
    val costFn = costFunFor[BFloat16](batchSample)
    Autodiff.valueAndGrad(costFn)(params)

  val jitCalcGradients = jit(calcGradients)
  val jitAdamWUpdate = jitDonatingUnsafe(opt.update[GPT.Params[Float32], Float32])

  def gradientDescentStep(
      runningBatchSamples: List[BatchSample],
      state: TrainingState
  ): TrainingState =
    val paramsF16 = state.params.asFloats(VType[BFloat16])

    val initialGrads = state.params.map([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, Float32]) => Tensor.like(x).fill(0f))
    val (accumulatedCosts, accumulatedGrads) =
      runningBatchSamples.foldLeft((Tensor0(0f), initialGrads)):
        case ((accCosts, accGrads), batchSample) =>
          val (costs, grads) = jitCalcGradients(batchSample, paramsF16)

          val newAccCosts = accCosts + costs.asFloat32 / accumulationSteps
          val newAccGrads = accGrads ++ grads.value.asFloats(VType[Float32]) `//!` accumulationSteps

          (newAccCosts, newAccGrads)

    val (params, optState) = jitAdamWUpdate(
      Grad(accumulatedGrads).clipGlobalNorm(gradientClipNorm),
      state.params,
      state.optState
    )

    val scalarLoss = accumulatedCosts.item
    TrainingState(params, optState, Tensor0(scalarLoss))

  val jitGradientDescentStep = gradientDescentStep

  val initState = TrainingState(initParams, opt.init(initParams), Tensor0(-1f))

  def miniBatchGradientDescent(
      samples: Iterator[BatchSample],
      startState: TrainingState
  ): Iterator[TrainingState] =
    samples.grouped(accumulationSteps).scanLeft(startState):
      case (state, runningBatches) =>
        jitGradientDescentStep(runningBatches.toList, state)

  val trainTrajectory = miniBatchGradientDescent(trainStream, initState)

  val logger = TensorTreeCheckpointer.newIn("out/GPT-2-multi-device")

  val trainMonitor = Monitor.ConcatMonitor[TrainingState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(state => state.stepCost.item),
    Monitor.PerformanceMonitor(effectiveBatchSize),
    Monitor.ThroughputMonitor(effectiveBatchSize * contextExtent.size, unitName = "tokens"),
    Monitor.LearningRateMonitor(schedule)
  ))

  println("Training...")

  val finalState = trainTrajectory
    .drop(1)
    .tapEvery(1):
      case (state, _) =>
        println(trainMonitor.report(state.optState.step.item, state))
    .tapEvery(1_000):
      case (state, _) =>
        val step = state.optState.step.item
        println("-" * 30)
        println(s"Performing validation at step $step...")
        val params = state.params.asFloats(VType[BFloat16])
        val avgValLoss = (1 to numBatchesPerValidation).iterator
          .map:
            case _ =>
              val valBatch = valStream.next()
              jitCostFn(params, valBatch).asFloat32.item
          .sum / numBatchesPerValidation
        println(s"Validation cost $step: ${avgValLoss}")
        logger.save(state, step)
        println(s"Checkpoint saved")
        println("-" * 30)
    .after(1_000_000_000)
