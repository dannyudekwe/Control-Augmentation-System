function in = pfcResetFcn(in) 
% Reset function for PFC multi-agent RL example
% randomize reference signal
blk = sprintf('qCAS_v2019b/Reference');
h = 1;
in = setBlockParameter(in,blk,'Value',num2str(h));

end