function [matrixF, matrixH, R, Q, dt ,ct] = DNS(ft, mean_states, lambda, alpha, sigmas)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      As in the FKF notation::
%      y_t = c_t + Z_t * a_t + G_t * eps_t,
%      a_t+1 = *d_t* + T_t * a_T + H_t eta_t.
%   
%      As in ATSE:   
%      y_t = c_t + H_t * xi_t + w_t ,  w_t ~ N(0,R)
%      xi_t+1 = *d_t* + F * xi_t +  v_t.   ~N(0, Q) 
%   
%     Zt :: H_t     , Tt :: Ft,   GGt :: R,   HHt ::Q
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mat = [1,2,5,10,20,30];

matrixF = eye(4);
matrixF(1,1) = ft(1);
matrixF(2,2) = ft(2);
matrixF(3,3) = ft(3);
matrixF(4,4) = ft(4);


matrixH = Loadings_Joint(lambda, alpha, mat(1)); 
for i = 2:length(mat)
    matrixH = vertcat(matrixH, Loadings_Joint(lambda, alpha, mat(i)));
end
%R = eye(12);
v= [6 6 5 5 4 4 2 2 1 1 0.7 0.7]/1200;
v= [6 6 5 5 4 4 3 3 2 2 1 1]/1200;
%v=  [12:-1:1]/12;
R = diag(v);
%R = eye(12);
%R = zeros(12);

q = eye(4);
q(1,1) = sigmas(1);
q(2,2) = sigmas(2);
q(3,3) = sigmas(3);
q(4,4) = sigmas(4);

Q = q;
%Q = matrixF * q * matrixF';

% temp = eye(4);
% temp(1,1) = 1-ft(1);
% temp(2,2) = 1-ft(2);
% temp(3,3) = 1-ft(3);
% temp(4,4) = 1-ft(4);
dt = (eye(4) - matrixF) * mean_states';
%dt = temp * mean_states';
% dt = zeros(4,1);
ct = zeros(12,1);
end
