function Value = V_Prob(ev, he)
%   CALCULATES    Pr[Yt/St,Yt-1]
%   gives 12x12 but should be 1x1. 
% sw_ms :: http://econ.korea.ac.kr/~cjkim/MARKOV/programs/sw_ms.opt
% :: VAL=(1/SQRT(DET(HE)))*EXP(-0.5*EV'*INV(HE)*EV);

%Value=(1./sqrt(2*pi*det(he)))*exp(-0.5*ev'*inv(he)*ev);
%Value=(1/sqrt(det(he)))*exp(-0.5*ev'*inv(he)*ev);
Value=(1/sqrt(2*pi*det(he)))*exp(-0.5*ev'*(he\ev));
end

