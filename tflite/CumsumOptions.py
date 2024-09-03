# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class CumsumOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CumsumOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsCumsumOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def CumsumOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # CumsumOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CumsumOptions
    def Exclusive(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # CumsumOptions
    def Reverse(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def CumsumOptionsStart(builder):
    builder.StartObject(2)

def Start(builder):
    CumsumOptionsStart(builder)

def CumsumOptionsAddExclusive(builder, exclusive):
    builder.PrependBoolSlot(0, exclusive, 0)

def AddExclusive(builder, exclusive):
    CumsumOptionsAddExclusive(builder, exclusive)

def CumsumOptionsAddReverse(builder, reverse):
    builder.PrependBoolSlot(1, reverse, 0)

def AddReverse(builder, reverse):
    CumsumOptionsAddReverse(builder, reverse)

def CumsumOptionsEnd(builder):
    return builder.EndObject()

def End(builder):
    return CumsumOptionsEnd(builder)
