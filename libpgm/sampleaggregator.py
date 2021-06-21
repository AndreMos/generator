# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
This module provides tools for collecting and managing sets of samples generated by the library's sampling functions. By averaging a series of samples, the progam can approximate a joint probability distribution without having to do the exact calculations, which may be useful in large networks. 

'''


class SampleAggregator(object):
    '''
    This class is a machine for aggregating data from sample sequences. It contains the method *aggregate*.
    
    '''
    def __init__(self):
        self.seq = None
        '''The sequence inputted.'''
        self.avg = None
        '''The average of all the entries in *seq*, represented as a dict where each vertex has an entry whose value is a dict of {key, value} pairs, where each key is a possible outcome of that vertex and its value is the approximate frequency.'''


    def aggregate(self, samplerstatement):
        '''
        Generate a sequence of samples using *samplerstatement* and return the average of its results. 
        
        Arguments:
            1. *samplerstatement* -- The statement of a function (with inputs) that would output a sequence of samples. For example: ``bn.randomsample(50)`` where ``bn`` is an instance of the :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` class.
        
        This function stores the output of *samplerstatement* in the attribute *seq*, and then averages *seq* and stores the approximate distribution found in the attribute *avg*. It then returns *avg*. 
       
        Usage example: this would print the average of 10 data points::

            import json

            from libpgm.nodedata import NodeData
            from libpgm.graphskeleton import GraphSkeleton
            from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from libpgm.sampleaggregator import SampleAggregator

            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittestdict.txt")
            skel.load("../tests/unittestdict.txt")

            # topologically order graphskeleton
            skel.toporder()

            # load bayesian network
            bn = DiscreteBayesianNetwork(skel, nd)

            # build aggregator
            agg = SampleAggregator()

            # average samples
            result = agg.aggregate(bn.randomsample(10))

            # output
            print json.dumps(result, indent=2)

        '''
        
        # get sequence
        seq = samplerstatement
        
        # denominator
        denom = len(seq)
        
        output = dict()
        for key in list(seq[0].keys()):
            output[key] = dict()
            for trial in seq:
                keyss = list(output[key].keys())
                vall = trial[key]
                if (keyss.count(vall) > 0):
                    output[key][trial[key]] += 1
                else:
                    output[key][trial[key]] = 1
                    
            # normalize
            for entry in list(output[key].keys()):
                output[key][entry] = output[key][entry] / float(denom)
        
        self.seq = seq
        self.avg = output
        
        return output
