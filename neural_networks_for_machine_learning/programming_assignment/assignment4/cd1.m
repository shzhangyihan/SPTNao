function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data)
    hidden_prob = visible_state_to_hidden_probabilities(rbm_w, visible_data)
    hidden_state = sample_bernoulli(hidden_prob)
    ret = zeros(size(rbm_w))
    ret = ret + configuration_goodness_gradient(visible_data, hidden_state)
    visible_prob = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
    visible_recon = sample_bernoulli(visible_prob)
    hidden_prob = visible_state_to_hidden_probabilities(rbm_w, visible_recon)
    ret = ret - configuration_goodness_gradient(visible_recon, hidden_prob)
end
