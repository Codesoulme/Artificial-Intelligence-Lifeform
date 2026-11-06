"""Bounded-memory random access to complete encoded dialogue records."""
import json,operator
from array import array
from collections.abc import Sequence

class DialogueRows(Sequence):
    def __init__(self,path,length):
        self.offsets=array('Q');self.stream=open(path,'rb');offset=0
        try:
            for line in self.stream:
                row=json.loads(line)
                if len(row['tokens'])!=len(row['mask']):raise ValueError('Misaligned loss mask')
                if len(row['tokens'])<=length+1 and any(row['mask'][1:]):self.offsets.append(offset)
                offset+=len(line)
        except BaseException:
            self.stream.close();raise
    def __len__(self):return len(self.offsets)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        index=operator.index(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        self.stream.seek(self.offsets[index]);return json.loads(self.stream.readline())
    def close(self):self.stream.close()
    def __del__(self):
        if hasattr(self,'stream'):self.stream.close()
