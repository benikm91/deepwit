package deepwit.training

import scala.collection.mutable.ListBuffer
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TapEverySuite extends AnyFunSpec with Matchers:

  describe("Iterator.tapEvery"):

    it("fires at every n-th index but not at zero"):
      val seen = ListBuffer.empty[(String, Int)]
      Iterator.from(0).map(i => s"e$i").tapEvery(3)((t, id) => seen += ((t, id))).take(10).toList
      seen.toList shouldBe List(("e3", 3), ("e6", 6), ("e9", 9))

    it("passes the elements through unchanged"):
      val result = Iterator.from(0).tapEvery(2)((_, _) => ()).take(5).toList
      result shouldBe List(0, 1, 2, 3, 4)

    it("does not fire beyond what is consumed"):
      val seen = ListBuffer.empty[Int]
      Iterator.from(0).tapEvery(1)((_, id) => seen += id).take(3).toList
      seen.toList shouldBe List(1, 2)

  describe("Iterator.after"):

    it("counts from zero, so the first element is the state after no steps"):
      Iterator.from(0).after(0) shouldBe 0

    it("returns the element that many steps in"):
      Iterator.from(0).after(3) shouldBe 3

    it("advances the iterator past what it returns"):
      val trajectory = Iterator.from(0)
      trajectory.after(3) shouldBe 3
      trajectory.next() shouldBe 4

    it("throws when the iterator ends first"):
      a[NoSuchElementException] should be thrownBy Iterator(0, 1).after(5)

    it("rejects a negative number of steps"):
      an[IllegalArgumentException] should be thrownBy Iterator.from(0).after(-1)
