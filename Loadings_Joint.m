function MatrixH = Loadings_Joint(lambda, alpha, time)
%   Dynamic Nelson Siegel loadings for the Joint model
%   Detailed explanation goes here

row1 = [1, (1 - exp(-lambda * time))/(lambda * time), (1 - exp(-lambda * time))/(lambda * time) - exp(-lambda * time), 0];
row2 = [0, alpha * ((1 - exp(-lambda * time))/(lambda * time)), alpha * ((1 - exp(-lambda * time))/(lambda * time) - exp(-lambda * time)), 1];
MatrixH = cat(1,row1, row2);




end


