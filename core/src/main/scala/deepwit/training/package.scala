package deepwit.training

/** Runs `f` on every `n`-th element of a training trajectory, passing the element and its index. */
extension [T](it: Iterator[T])

  def tapEvery(n: Int)(f: (T, Int) => Unit): Iterator[T] =
    it
      .zipWithIndex
      .tapEach: (t, id) =>
        if id > 0 && id % n == 0 then f(t, id)
      .map(_._1)

  /** The state after n iterations: Advances the iterator n steps and returns the resulting element */
  def after(n: Int): T =
    require(n >= 0, s"A number of steps must not be negative, but was $n.")
    it.drop(n).next()
