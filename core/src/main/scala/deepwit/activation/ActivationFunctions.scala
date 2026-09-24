package deepwit.activation

import dimwit.*

def sigmoid[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = t.sigmoid

def relu[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = t.relu

def gelu[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = t.gelu

def softmax[L: Label, V: IsFloating](t: Tensor1[L, V]): Tensor1[L, V] = t.softmax(Axis[L])
