"""
In this file, you should implement the forward calculation of the basic RNN model and the RNN model with GRUs. 
Please use the provided interface. The arguments are explained in the documentation of the two functions.
"""

import numpy as np
from scipy.special import expit as sigmoid

def rnn(wt_h, wt_x, bias, init_state, input_data):
    """
    RNN forward calculation.
    inputs:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    outputs:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """

    outputs = None
    final_state = None
    
    ##################################################################################################
    # Please implement the basic RNN here. You don't need to considier computational efficiency.     #
    ##################################################################################################


    return outputs, final_state


def gru(wtu_h, wtu_x, biasu, wtr_h, wtr_x, biasr, wtc_h, wtc_x, biasc, init_state, input_data):
    """
    RNN forward calculation.

    inputs:
        wtu_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for u gate
        wtu_x: shape [input_size, hidden_size], weight matrix for input transformation for u gate
        biasu: shape [hidden_size], bias term for u gate
        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        wtc_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for candicate
               hidden state calculation
        wtc_x: shape [input_size, hidden_size], weight matrix for input transformation for candicate
               hidden state calculation
        biasc: shape [hidden_size], bias term for candicate hidden state calculation
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    outputs:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """

    outputs = None
    final_state = None
    ##################################################################################################
    # Please implement an RNN with GRU here. You don't need to considier computational efficiency.   #
    ##################################################################################################
       
    
    return outputs, final_state

