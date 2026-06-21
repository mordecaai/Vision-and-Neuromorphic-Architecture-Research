import pyrtl

def stdp_synapse(pre_spike: pyrtl.WireVector, post_spike: pyrtl.WireVector, weight_bitwidth: int = 8, timer_bitwidth: int = 5, prefix: str = "syn") -> pyrtl.WireVector:
    """
    An adaptive STDP Synapse.
    Outputs the weighted current when a dynamic vision sensor (DVS) pixel fires (pre_spike).
    Adapts its own weight based on the timing between pre_spike and post_spike.
    LIF Neuron action potential (post-spike) feeds back into corresponding STDP synapses.

    :param pre_spike: 1-bit WireVector from the upstream DVS camera pixel. 
                      A '1' indicates an edge was detected, opening the synapse lane.
    :param post_spike: 1-bit WireVector feedback loop from the corresponding downstream LIF Neuron. 
                       A '1' indicates the LIF neuron just fired, triggering the STDP 
                       logic to evaluate the temporal usefulness of its corresponding pre-synaptic
                        neuron and update the internal weight.
    :param weight_bitwidth: Integer defining the physical size (number of flip-flops) of the 
                     internal weight register. An 8-bit width  
                     bounds the memory values between 0 and 255.
    :param timer_bitwidth: Integer defining the physical size (number of flip-flops) of the 
                    inernal timer register. A 5-bit bound is enough
                    for the 20 cycle timeout max.
    :param prefix: String prepended to internal register names (e.g., 'syn_y2_x5'). 
                   Useful for maintaining spatial locality and preventing duplicate 
                   wire naming errors during hardware synthesis.
    :return: WireVector of size 'bitwidth' representing the injected current. 
             Outputs the stored weight if pre_spike is 1; outputs 0 otherwise.
    """
    timer = pyrtl.Register(bitwidth=timer_bitwidth, name=f"{prefix}_timer")
    weight = pyrtl.Register(bitwidth=weight_bitwidth, name=f"{prefix}_weight")

    timeout = pyrtl.Const(20, timer_bitwidth)
    
    # Dynamic limits based on the bitwidth parameter
    max_weight_val = (2 ** weight_bitwidth) - 1
    max_weight = pyrtl.Const(max_weight_val, weight_bitwidth)
    zero_weight = pyrtl.Const(0, weight_bitwidth)

    # 1. Timer Logic
    timer.next <<= pyrtl.select(
        pre_spike, 
        pyrtl.Const(0, timer_bitwidth), 
        pyrtl.select(timer < timeout, timer + pyrtl.Const(1, timer_bitwidth), timer)
    )

    # Determine whether spike is within window of biological relevance
    valid_window = timer < timeout
    trigger_update = post_spike & valid_window

    # 2. Piecewise Reward Tiers (Simplified Sigmoid)
    tier1_limit = pyrtl.Const(5, timer_bitwidth)
    tier2_limit = pyrtl.Const(15, timer_bitwidth)

    is_tier1 = timer < tier1_limit
    is_tier2 = (timer >= tier1_limit) & (timer < tier2_limit)
    is_late = timer >= tier2_limit

    # 3. Math
    add_amount = pyrtl.select(
        is_tier1, pyrtl.Const(3, weight_bitwidth),
        pyrtl.select(is_tier2, pyrtl.Const(1, weight_bitwidth), zero_weight)
    )
    
    sub_amount = pyrtl.select(is_late, pyrtl.Const(1, weight_bitwidth), zero_weight)

    # 4. Overflow / Underflow Safety
    safe_to_add = weight <= (max_weight - add_amount)
    safe_to_sub = weight >= sub_amount

    # Apply limits
    added_w = pyrtl.select(safe_to_add, weight + add_amount, max_weight)
    final_new_weight = pyrtl.select(safe_to_sub, added_w - sub_amount, zero_weight)

    # 5. The Final Update 
    weight.next <<= pyrtl.select(trigger_update, final_new_weight, weight)

    # 6. Output
    weighted_current = pyrtl.select(pre_spike, weight, zero_weight)

    return weighted_current
