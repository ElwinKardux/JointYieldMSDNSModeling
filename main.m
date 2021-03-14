%--------------------------------------------------------------------------
%   Author: Elwin Kardux
%   Date: 2021-02-27
%   
%   Markov Switching Dynamic Nelson Siegel:: Kim Filter
%
%   
%   TODO: -Get results, smoothed states?:: function given in Kim gauss codes
%         -Plot results as in current python files? not necessary, can also convert to csv file as we did in R.
%         -Check if we get sensible results with the Kalman Filter. 
%              Compare with the papers.
%         -Dont do diagonal F_t in the 4states!
%         -Check whether Arbitrage free part is similar for 4 factors + Markov Switching, so
%         far it seems so! Then we also extend it with the arbitrage-freeness! 
%
%::  Expected: -diagonals F_t: between 0.7 and 1.8?
%::            -lambda single: 0.53,
%::            -sigmas: 
%--------------------------------------------------------------------------
%% 1) Prepare Data 
%load /Data/denom.csv;
%dir('*/Data/*.csv')
z_denom = readtable('denom interpolated.csv','HeaderLines',1);
z_deinf = readtable('deinf interpolated.csv', 'HeaderLines',1);
denom = table2array(z_denom(:,2:end));
deinf = table2array(z_deinf(:,2:end));
dedates = table2array(z_denom(:,1));
st = "2000-01-31";
et =  "2020-12-31";
mat = [1, 2, 5, 10, 20, 30];

empL_nom = [denom(:,1) + denom(:,3) + denom(:,6)]/3;
empSlope = -(  (denom(:,6) + deinf(:,6))/2 - (denom(:,1) + deinf(:,1))/2);
empCurv = (2*((denom(:,3) + deinf(:,3))/2) - (denom(:,1) + denom(:,6) + deinf(:,1) + deinf(:,6))/2);
empL_inf = (deinf(:,1) + deinf(:,3) + deinf(:,6))/3;

dejoi = horzcat(denom(:, 1), deinf(:, 1)); 
for i = 2:length(mat)
  dejoi = horzcat(dejoi, denom(:, i), deinf(:, i));
end

y252 = dejoi;
RMSE = zeros(12,10);
AIC = zeros(1,10);
BIC = zeros(1,10);

speed = 1e+7;
speed2 = 1e-7;
%==========================================================================
%      As in ATSE:   
%      y_t =     H_t * xi_t +  w_t ,  w_t ~ N(0,R)
%      xi_t+1 =  F   * xi_t +  v_t.   v_t ~ N(0,Q) 
%==========================================================================

%% Standard model: no markov switching component but with kim filter?:: similar to kalman filter

% Parameter initialization: according to estimates of CDR. joint AFNS.
ft = [0.7, 0.7, 0.7, 0.7]; %: diagonals in F_t
lambda = 0.15;
alpha = 0.7072; % from CDR (2010)
S0_pr = 0.96;                        %expansion
S1_pr = 0.90;                        %recession
sigmas = [0.5, 0.5, 0.5, 0.5];

mu_states = [6.22, -1.50, 4.4]; %from paper: DNS estimates.
%[6.21592195946671,-1.50372152538060,4.39682231002382,5.49804805765792]

param_vector = [0.7, 0.7, 0.7, 0.7,...  %ft
    0.05, alpha, 1.0, 0,...     
    0.5, 0.5, 0.5, 0.5,...    %sigmas
    0, 0, 0, 0];

%  @===================================================================@
%  @======================== Constrained ==============================@
%  @===================================================================@
% Lower bound for the parameter vector: Sigma > 0
lb = [-Inf;-Inf;-Inf;-Inf;...                   % ft   
      -Inf;  -Inf;   0;   0;...                   %lambda, alpha, S0_pr, S1_pr 
         0;   0;   0;   0;...                   %sigmas
      -Inf;-Inf;-Inf;-Inf];                     %mu_states
    
% Upper bound:: P_pr, Q_Pr <=1
ub = [1;1; 1; 1;... ft should be <1 for convergence.
      Inf;  Inf;   1;   1;...
      Inf;Inf; Inf; Inf;...
      Inf;Inf; Inf; Inf]; 

[negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred] = NegativeLogLikeMS(param_vector, y252);
%[matrixF, matrixH, R, Q] = DNS(ft, lambda, alpha, sigmas);


clearvars options
options  =  optimset('fmincon'); % This sets the options at their standard values
options  =  optimset(options , 'MaxFunEvals' , speed); % extra iterations
options  =  optimset(options , 'MaxIter'     , speed);
options  =  optimset(options , 'TolFun'      , speed2); % extra precision
options  =  optimset(options , 'TolX'        , speed2); % extra precision
[par_con, LogL_con, ~, ~,~,~, Hessian_DNS_con ]=fmincon('NegativeLogLikeMS', param_vector,[],[],[],[],lb,ub,[],options,y252);
par_con'
std_error_DNS = sqrt(diag(inv(Hessian_DNS_con)))

%:: Currently gives: [0.82 ,0.73,0.71, 0.83,0.52 ,0.71 ,0.95 ,0.80 ,0.301,0.300164031967288,0.299737175695198,0.288717922227763,6.21579478140029,-1.50380376577379,4.39685047777934,5.49787426258193]

%  @===================================================================@
%  @======================= Unconstrained =============================@
%  @===================================================================@
% clearvars options
% options = optimset('fminunc');
% options = optimset(options , 'MaxFunEvals', speed);
% options = optimset(options , 'MaxIter' ,speed);
% options = optimset(options , 'TolFu' , speed2);
% options = optimset(options , 'TolX' , speed2);
% [par_uncon, LogL_uncon, ~, ~, Hessian_DNS_uncon ]=fminunc('NegativeLogLikeMS', param_vector,options, y252);


%% Forecasting: 780 months ahead
[negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE_, AIC_, BIC_, xi_smooth_DNS, pr0_smooth_DNS] = NegativeLogLikeMS(par_con, y252);
[F, H, R, Q, dt, ct] = DNS(par_con(1:4), par_con(13:16), par_con(5), par_con(6), par_con(9:12));
% in-sample:
RMSE(:,1) = RMSE_';
AIC(1,1) = AIC_;
BIC(1,1) = BIC_;
 
% Quick Check!
figure
plot(xi_smooth_DNS(1,:))
hold on
plot(empL_nom);

figure
plot(xi_smooth_DNS(2,:))
hold on
plot(empSlope)

figure
plot(xi_smooth_DNS(3,:)/2)
hold on
plot(empCurv)

figure
plot(xi_smooth_DNS(4,:)-1)
hold on
plot(empL_inf)

MC = [S0_pr 1-S0_pr; 1-S1_pr  S1_pr];
MarkovPath = simulate(dtmc(MC),780); % simulated States 780 times in the future. 

xi_forecast_con = zeros(4,780);
yield_forecast_con = zeros(12,780);

xi_forecast_con(:,1) = dt + F*xi_0(:,end);
yield_forecast_con(:,1) = ct + H*xi_forecast_con(:,1);

for i=2:780
    xi_forecast_con(:,i) = dt + F*xi_forecast_con(:,i-1) + normrnd(0,Q) * ones(4,1);
    yield_forecast_con(:,i) = ct + H*xi_forecast_con(:,i);
end

filename = 'forecasts\DNS_de_simulate_con.csv';
writematrix(yield_forecast_con',filename)


% for unconstrained: 
% [negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE_, AIC_, BIC_, xi_smooth, pr0_smooth] = NegativeLogLikeMS(par_uncon, y252);
% [F, H, R, Q, dt, ct] = DNS(par_uncon(1:4), par_uncon(13:16), par_uncon(5), par_uncon(6), par_uncon(9:12));
% % in-sample:
% RMSE(:,2) = RMSE_';
% AIC(1,2) = AIC_;
% BIC(1,2) = BIC_;
% 
% xi_forecast_uncon = zeros(4,780);
% yield_forecast_uncon = zeros(12,780);
% 
% xi_forecast_uncon(:,1) = dt + F*xi_0(:,end);
% yield_forecast_uncon(:,1) = ct + H*xi_forecast_uncon(:,1);
% 
% for i=2:780
%     xi_forecast_uncon(:,i) = dt + F*xi_forecast_uncon(:,i-1) + normrnd(0,Q) * ones(4,1);
%     yield_forecast_uncon(:,i) = ct + H*xi_forecast_uncon(:,i);
% end
% 
% filename = 'forecasts\DNS_de_simulate_uncon.csv';
% writematrix(yield_forecast_uncon',filename)


%% %===============================================================
%==========================================================================
%==========================================================================
%==========================================================================
%       Markov Switching Lambda
%==========================================================================
%==========================================================================
%==========================================================================
%   This gives regime switching in the H matrix. 
%   2 lambda's to estimate, one for each regime. 
%       in MS-DNS paper with 3 factors: lambda_0 = 0.153,  lambda_1 = 0.055. 
%       However, 4 factors gives different lambda! 
%
%==========================================================================

ft = [0.8, 0.8, 0.8, 0.8]; %: diagonals in F_t
lambda_0 = 0.08;
alpha = 0.7072;
S0_pr = 0.96;                        %expansion
S1_pr = 0.90;                        %recession
%sigmas = [0.00446, 0.00755, 0.029, 0.3];
sigmas = [0.36, 0.62, 0.92, 0.5];
mu_states = [6.22, -1.50, 4.4, 5.50]; %from paper: DNS estimates.
lambda_1 = 0.08;


param_vector_MSL = [0.7, 0.7, 0.7, 0.7,...  %ft
    0.10, alpha, 0.97, 0.92,...     
    0.3, 0.3, 0.3, 0.3,...    %sigmas
    0, 0, 0, 0, ...
    0.40];

[negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred] = NegLogLikeLambda(param_vector_MSL, y252);


%  @===================================================================@
%  @========================= Constrained =============================@
%  @===================================================================@
% Lower bound for the parameter vector: Sigma > 0
lb_MSL =[0;   0;   0;   0;...               % ft   
        0;  -1;   0;   0;...                   %lambda, alpha, S0_pr, S1_pr 
         0;   0;   0;   0;...                   %sigmas
      -Inf;-Inf;-Inf;-Inf; 0];                  %mu_states
    
% Upper bound:: P_pr, Q_Pr <=1
ub_MSL = [1  ;  1;   1;   1;...
          1;  1;   1;   1;...
          Inf;Inf; Inf; Inf;...
          Inf;Inf; Inf; Inf; 1]; 

clearvars options
options  =  optimset('fmincon'); % This sets the options at their standard values
options  =  optimset(options , 'MaxFunEvals' , speed); % extra iterations
options  =  optimset(options , 'MaxIter'     , speed);
options  =  optimset(options , 'TolFun'      , speed2); % extra precision
options  =  optimset(options , 'TolX'        , speed2); % extra precision

[parMSL_con, LogLMSL_con, ~, ~,~,~, Hessian_MSL_con ]=fmincon('NegLogLikeLambda', param_vector_MSL,[],[],[],[],lb_MSL,ub_MSL,[],options,y252);
parMSL_con'
std_error_MSLambda = sqrt(diag(inv(Hessian_MSL_con)))
%  @===================================================================@
%  @======================= Unconstrained =============================@
%  @===================================================================@
% clearvars options
% options = optimset('fminunc');
% options = optimset(options , 'MaxFunEvals' , speed);
% options = optimset(options , 'MaxIter' ,speed);
% options = optimset(options , 'TolFu' , speed2);
% options = optimset(options , 'TolX' , speed2);
% 
% [parMSL_uncon, LogLMSL_uncon, ~, ~, Hessian_MSL_uncon ]=fminunc('NegLogLikeLambda', param_vector_MSL,options, y252);

%% Forecasting: 780 months ahead
seed = 11;
rng(seed)
normrnd(0,1) %check

S0_pr = parMSL_con(7);
S1_pr = parMSL_con(8);
MC = [S0_pr 1-S0_pr; 1-S1_pr  S1_pr];
MarkovPath = simulate(dtmc(MC),780); % simulated States 780 times in the future. 

[negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE_, AIC_, BIC_, xi_smooth_MSL, pr0_smooth_MSL] = NegLogLikeLambda(parMSL_con, y252);
[F, H_0, R, Q, dt, ct] = DNS(parMSL_con(1:4), parMSL_con(13:16), parMSL_con(5), parMSL_con(6), parMSL_con(9:12));
[~, H_1, ~, ~, ~, ~] = DNS(parMSL_con(1:4), parMSL_con(13:16), parMSL_con(17), parMSL_con(6), parMSL_con(9:12));

figure
plot(xi_smooth_MSL(1,:))
hold on
plot(empL_nom);

figure
plot(xi_smooth_MSL(2,:))
hold on
plot(empSlope)

figure
plot(xi_smooth_MSL(3,:))
hold on
plot(empCurv)

figure
plot(xi_smooth_MSL(4,:)-1)
hold on
plot(empL_inf)

RMSE(:,3) = RMSE_';
AIC(1,3) = AIC_;
BIC(1,3) = BIC_;
dt = dt - dt;


%constrained forecasts
xi_forecast_MSL_con = zeros(4,780,100);
yield_forecast_MSL_con = zeros(12,780,100);

MarkovPath = zeros(100,781);

for path= 1:100
    MarkovPath(path,:) = simulate(dtmc(MC),780);
    xi_forecast_MSL_con(:,1,path) = dt + F*xi_0(:,end);
    yield_forecast_MSL_con(:,1,path) = y252(252,:)' +ct + H*xi_forecast_MSL_con(:,1);
    for i=2:780
        if MarkovPath(path,i)==1
            xi_forecast_MSL_con(:,i,path) = dt + F*xi_forecast_MSL_con(:,i-1,path)+ normrnd(0,Q) * ones(4,1)*10;
            yield_forecast_MSL_con(:,i,path) = y252(252,:)' +ct + H_0*xi_forecast_MSL_con(:,i,path)+ normrnd(0,R) * ones(12,1)*1;
        else
            xi_forecast_MSL_con(:,i,path) = dt + F*xi_forecast_MSL_con(:,i-1,path) + normrnd(0,Q) * ones(4,1)*10;
            yield_forecast_MSL_con(:,i,path) = y252(252,:)' +ct + H_1*xi_forecast_MSL_con(:,i,path) + normrnd(0,R) * ones(12,1)*1;    
        end
        
    end
end

yield_forecasts_mean_MSL_con = mean(yield_forecast_MSL_con,3);
yield_forecasts_mean_MSL_con(:,780)
filename = 'forecasts\MSL_DNS_de_simulate_con.csv';
writematrix(yield_forecasts_mean_MSL_con',filename)


% unconstrained forecasts:
% [negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE_, AIC_, BIC_] = NegLogLikeLambda(parMSL_uncon, y252);
% [F, H_0, R, Q, dt, ct] = DNS(parMSL_uncon(1:4), parMSL_uncon(13:16), parMSL_uncon(5), parMSL_uncon(6), parMSL_uncon(9:12));
% [~, H_1, ~, ~, ~, ~] = DNS(parMSL_uncon(1:4), parMSL_uncon(13:16), parMSL_uncon(17), parMSL_uncon(6), parMSL_uncon(9:12));
% % in-sample:
% RMSE(:,4) = RMSE_';
% AIC(1,4) = AIC_;
% BIC(1,4) = BIC_;

% Unconstrained forecasts
% S0_pr = parMSL_uncon(7);
% S0_pr = 0.95
% S1_pr = parMSL_uncon(8);
% MC = [S0_pr 1-S0_pr; 1-S1_pr  S1_pr];
% MarkovPath = simulate(dtmc(MC),780); % simulated States 780 times in the future. 
% 
% xi_forecast_MS_uncon = zeros(4,780,100);
% yield_forecast_MS_uncon = zeros(12,780,100);
% 
% MarkovPath = zeros(100,781);
% 
% for path= 1:100
%     MarkovPath(path,:) = simulate(dtmc(MC),780);
%     xi_forecast_MS_uncon(:,1,path) = dt + F*xi_0(:,end);
%     yield_forecast_MS_uncon(:,1,path) =  y252(252,:)' + H*xi_forecast_MS_uncon(:,1);
%     for i=2:780
%         if MarkovPath(path,i)==1
%             xi_forecast_MS_uncon(:,i,path) = dt + F*xi_forecast_MS_uncon(:,i-1,path) + normrnd(0,Q) * ones(4,1);
%             yield_forecast_MS_uncon(:,i,path) = y252(252,:)' + ct + H_0*xi_forecast_MS_uncon(:,i,path);
%         else
%             xi_forecast_MS_uncon(:,i,path) = dt + F*xi_forecast_MS_uncon(:,i-1,path) + normrnd(0,Q) * ones(4,1);
%             yield_forecast_MS_uncon(:,i,path) =y252(252,:)' + ct + H_1*xi_forecast_MS_uncon(:,i,path);    
%         end
%         
%     end
% end
% 
% yield_forecasts_mean_MS_uncon = mean(yield_forecast_MS_uncon,3);
% 
% filename = 'forecasts\MSL_DNS_de_simulate_uncon.csv';
% writematrix(yield_forecasts_mean_MS_uncon',filename)

%% ===============================================================
%==========================================================================
%==========================================================================
%==========================================================================
%==========================================================================
%       Markov Switching Sigmas
%==========================================================================
%==========================================================================
%==========================================================================
%==========================================================================
%   This gives regime switching in the Q matrix. 
%   8 sigmas to estimate instead of 4. 
%       
%==========================================================================


ft = [0.8, 0.8, 0.7, 0.8]; %: diagonals in F_t
lambda = 0.05;
alpha = 0.8072;
S0_pr = 0.96;                        %expansion
S1_pr = 0.93;                        %recession
sigmas_0 = [0.5, 1.22, 1.87, 0.6];
mu_states = [6.22, -1.50, 4.4]; %from paper: DNS estimates.
sigmas_1 = [0.26, 0.33, 0.61, 0.30];  

% param_vector_MSSigmas = [0.8, 0.8, 0.7, 0.8,...  %ft
%     0.05, alpha, 0.98, 0.91,...     
%     0.26, 0.33, 0.61, 0.26,...    %sigmas
%     6.85, 0.83, -0.66, 5.50, ...
%     0.4, 0.8, 1.17, 0.4];

param_vector_MSSigmas = [0.7, 0.7, 0.7, 0.7,...  %ft
    0.05, alpha, 0.96, 0.91,...     
    0.26, 0.33, 0.61, 0.26,...    %sigmas
    6.85, 0.83, -0.66, 5.50, ...
    0.4, 0.8, 1.17, 0.4];

%  @===================================================================@
%  @========================= Constrained =============================@
%  @===================================================================@
% Lower bound for the parameter vector: Sigma > 0
lb_MSSigmas = [0;0 ;0 ;0;...          % ft   
      -Inf;  -1;   0.01;   0.01;...                   %lambda, alpha, S0_pr, S1_pr 
         0;   0;   0;   0;...                   %sigmas_0
      -Inf;-Inf;-Inf;-Inf;...                   %mu_states
        0;   0;   0;   0];                      %sigmas_1                
    
% Upper bound:: P_pr, Q_Pr <=1
ub_MSSigmas = [1;1; 1; 1;...
      1;  1;   1;   1;...
      Inf;Inf; Inf; Inf;...
      Inf;Inf; Inf; Inf;...
      Inf;Inf; Inf; Inf]; 

%[negLogLike,  LL] = NegLogLikeLambda(param_vector_MSL, y252);
%[matrixF, matrixH, R, Q] = DNS(ft, lambda, alpha, sigmas);

[negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred] = NegLogLikeSigmas(param_vector_MSSigmas, y252);
clearvars options
options  =  optimset('fmincon'); % This sets the options at their standard values
options  =  optimset(options , 'MaxFunEvals' , speed); % extra iterations
options  =  optimset(options , 'MaxIter'     , speed);
options  =  optimset(options , 'TolFun'      , speed2); % extra precision
options  =  optimset(options , 'TolX'        , speed2); % extra precision

[parMSSigma_con, LogLMSSigma_con, ~, ~,~,~, Hessian_MSSigma_con ]=fmincon('NegLogLikeSigmas', param_vector_MSSigmas,[],[],[],[],lb_MSSigmas,ub_MSSigmas,[],options,y252);
parMSSigma_con'
std_error_MSSigma = sqrt(diag(inv(Hessian_MSSigma_con)))
%  @===================================================================@
%  @======================= Unconstrained =============================@
%  @===================================================================@
% clearvars options
% options = optimset('fminunc');
% options = optimset(options , 'MaxFunEvals' ,speed);
% options = optimset(options , 'MaxIter' ,speed);
% options = optimset(options , 'TolFu' , speed2);
% options = optimset(options , 'TolX' , speed2);
% 
% [parMSSigma_uncon, LogL_MSSigma_uncon, ~, ~, Hessian_MSSigma_uncon ]=fminunc('NegLogLikeSigmas', param_vector_MSSigmas,options, y252);


%% Forecasting: 780 months ahead
seed = 11;
rng(seed)
normrnd(0,1)

S0_pr = parMSSigma_con(7) +0.16;
S1_pr = parMSSigma_con(8) +0.35;
MC = [S0_pr 1-S0_pr; 1-S1_pr  S1_pr];
MarkovPath = simulate(dtmc(MC),780); % simulated States 780 times in the future. 

[negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE_, AIC_, BIC_, xi_smooth_MSS, pr0_smooth_MSS] = NegLogLikeSigmas(parMSSigma_con, y252);
[F, H, R, Q_0, dt, ct] = DNS(parMSSigma_con(1:4), parMSSigma_con(13:16), parMSSigma_con(5), parMSSigma_con(6), parMSSigma_con(9:12));
[~, ~, ~, Q_1,  ~,  ~] = DNS(parMSSigma_con(1:4), parMSSigma_con(13:16), parMSSigma_con(5), parMSSigma_con(6), parMSSigma_con(17:20));
% in-sample:
RMSE(:,5) = RMSE_';
AIC(1,5) = AIC_;
BIC(1,5) = BIC_;

% quick check!
figure
plot(xi_smooth(1,:))
hold on
plot(empL_nom);

figure
plot(xi_smooth(2,:))
hold on
plot(empSlope)

figure
plot(xi_smooth(3,:))
hold on
plot(empCurv)

figure
plot(xi_smooth(4,:)-1)
hold on
plot(empL_inf)

%slightly adjust dt to avoid going to -20% nominal/real yields.
dt(4) = dt(1);
%dt = dt + [0.0050; -dt(2)-0.0008; 0.002; 0.005];

dt = dt - dt;

dt = [0.010; 0.0001; 0.001; 0.010]; 
%dt_2 = [0.16; 0.0002; 0.008; 0.16];

xi_forecast_MSSigma_con = zeros(4,780,num_paths);
yield_forecast_MSSigma_con = zeros(12,780,num_paths);

xi_forecast_Italy = zeros(4,780,num_paths);
yield_forecast_Italy = zeros(12,780,num_paths);

italy_spread_mean = [0.59 0.59 0.7 0.7  1.03  1.03  1.25  1.25  1.24  1.24  1.21  1.21];
MarkovPath = zeros(num_paths,781);

for path= 1:num_paths
    MarkovPath(path,:) = simulate(dtmc(MC),780);
    xi_forecast_MSSigma_con(:,1,path) = dt + F*xi_0(:,end);
    yield_forecast_MSSigma_con(:,1,path) =  H*xi_forecast_MSSigma_con(:,1, path); %y252(252,:)'+
    yield_forecast_Italy(:,1,path)= yield_forecast_MSSigma_con(:,1,path) + italy_spread_mean';
    
    for i=2:780
        if MarkovPath(path,i)==1 % state 0
            xi_forecast_MSSigma_con(:,i,path) = dt + F*xi_forecast_MSSigma_con(:,i-1,path)+ normrnd(0,Q_0) * ones(4,1)*10;%4   %+ normrnd(0,eye(4))* ones(4,1)*0.01; %;
            yield_forecast_MSSigma_con(:,i,path) = ct + H*xi_forecast_MSSigma_con(:,i,path)+ normrnd(0,R) * ones(12,1)*20;%6
            
            xi_forecast_Italy(:,i,path) = xi_forecast_MSSigma_con(:,i,path) + normrnd(0, diag([0.2; 0.5; 0.5; 0.2]))*ones(4,1);
            yield_forecast_Italy(:,i,path) = yield_forecast_MSSigma_con(:,i,path) +  normrnd(italy_spread_mean', ones(12,1)*0.1);
        else % state 1
            xi_forecast_MSSigma_con(:,i,path) = dt + F*xi_forecast_MSSigma_con(:,i-1,path) +normrnd(0,Q_1) * ones(4,1)*10;%4% normrnd(0,eye(4))* ones(4,1)*0.3; %
            yield_forecast_MSSigma_con(:,i,path) =  ct + H*xi_forecast_MSSigma_con(:,i,path) + normrnd(0,R) * ones(12,1)*20;%5
            
            xi_forecast_Italy(:,i,path) = xi_forecast_MSSigma_con(:,i,path) + normrnd(0, diag([0.2; 0.5; 0.5; 0.2]))*ones(4,1);
            yield_forecast_Italy(:,i,path) = yield_forecast_MSSigma_con(:,i,path) +   normrnd(italy_spread_mean', ones(12,1)*0.1);
       
        end
        
    end
end

yield_forecasts_mean_MSSigma_con = mean(yield_forecast_MSSigma_con,3);
yield_forecasts_mean_MSSigma_con(:,780)
yield_forecast_Italy_mean = mean(yield_forecast_Italy,3);

filename = 'forecasts\MSSigma_DNS_de_simulate_con.csv';
writematrix(yield_forecasts_mean_MSSigma_con',filename)
filename = 'forecasts\MSSigma_Italy.csv';
writematrix(yield_forecast_Italy_mean',filename)


% Unconstrained forecasts
%ft, mean_states, lambda, alpha, sigmas
% [negLogLike,  LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE_, AIC_, BIC_] = NegLogLikeSigmas(parMSSigma_uncon, y252);
% [F, H_0, R, Q_0, dt, ct] = DNS(parMSSigma_uncon(1:4), parMSSigma_uncon(13:16), parMSSigma_uncon(5), parMSSigma_uncon(6), parMSSigma_uncon(9:12));
% [~, ~, ~, Q_1, ~, ~] = DNS(parMSSigma_uncon(1:4), parMSSigma_uncon(13:16), parMSSigma_uncon(5), parMSSigma_uncon(6), parMSSigma_uncon(17:20));
% % in-sample:
% RMSE(:,6) = RMSE_';
% AIC(1,6) = AIC_;
% BIC(1,6) = BIC_;
% 
% S0_pr = parMSSigma_uncon(7);
% S1_pr = parMSSigma_uncon(8);
% MC = [S0_pr 1-S0_pr; 1-S1_pr  S1_pr];
% MarkovPath = simulate(dtmc(MC),780); % simulated States 780 times in the future. 
% 
% xi_forecast_MSSigma_uncon = zeros(4,780,100);
% yield_forecast_MSSigma_uncon = zeros(12,780,100);
% 
% MarkovPath = zeros(100,781);
% 
% 
% for path= 1:100
%     MarkovPath(path,:) = simulate(dtmc(MC),780);
%     xi_forecast_MSSigma_uncon(:,1,path) = dt + F*xi_0(:,end);
%     yield_forecast_MSSigma_uncon(:,1,path) = ct + H*xi_forecast_MSSigma_uncon(:,1);
%     for i=2:780
%         if MarkovPath(path,i)==1
%             xi_forecast_MSSigma_uncon(:,i,path) = dt + F*xi_forecast_MSSigma_uncon(:,i-1,path) + normrnd(0,Q_0) * ones(4,1);
%             yield_forecast_MSSigma_uncon(:,i,path) = ct + H*xi_forecast_MSSigma_uncon(:,i,path);
%         else
%             xi_forecast_MSSigma_uncon(:,i,path) = dt + F*xi_forecast_MSSigma_uncon(:,i-1,path) + normrnd(0,Q_1) * ones(4,1);
%             yield_forecast_MSSigma_uncon(:,i,path) = ct + H*xi_forecast_MSSigma_uncon(:,i,path);    
%         end
%         
%     end
% end
% 
% yield_forecasts_mean_MSSigma_uncon = mean(yield_forecast_MSSigma_uncon,3);
% 
% filename = 'forecasts\MSSigma_DNS_de_simulate_uncon.csv';
% writematrix(yield_forecasts_mean_MSSigma_uncon',filename)

%% Smooth Estimate Plots 
time = dedates;

figure(gcf);
plot(time,xi_smooth_DNS(1,:),'SeriesIndex',1,'LineStyle', '-.', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSL(1,:),'SeriesIndex',5,'LineStyle', '--', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSS(1,:),'SeriesIndex',7, 'LineStyle', ':', 'LineWidth', 2);hold on;
plot(time, empL_nom+ 1.6, 'k', 'LineWidth',1.5); hold on;

plot([min(xlim()),max(xlim())],[0,0], 'k--', 'LineWidth', 0.2);
ax = gca; 
%txt1 = strcat("Ann. average return = ",annRet,"% ", annRetStd);
%txt2 = strcat("Max Drawdown =  ",MaxDD, "%");
%text(time(5),3.30,txt1,'FontSize',13)
%text(time(5),2.95,txt2,'FontSize',13)
ylim([-0.5, 8]);
xticks(time(1:60:end));
datetick('x', 'yyyy', 'keepticks')
ax.FontSize = 12;
%recessionplot;
set(gcf,'units','points','position',[10,10,800,200])


% Slope:: 
figure(gcf);
plot(time,xi_smooth_DNS(2,:),'SeriesIndex',1,'LineStyle', '-.', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSL(2,:),'SeriesIndex',5,'LineStyle', '--', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSS(2,:),'SeriesIndex',7,'LineStyle', ':', 'LineWidth', 2);hold on;
plot(time, empSlope-0.5, 'k', 'LineWidth',1.5); hold on;

plot([min(xlim()),max(xlim())],[0,0], 'k--', 'LineWidth', 0.2);
ax = gca; 
%txt1 = strcat("Ann. average return = ",annRet,"% ", annRetStd);
%txt2 = strcat("Max Drawdown =  ",MaxDD, "%");
%text(time(5),3.30,txt1,'FontSize',13)
%text(time(5),2.95,txt2,'FontSize',13)
ylim([-5.7, 1.0]);
xticks(time(1:60:end));
datetick('x', 'yyyy', 'keepticks')
ax.FontSize = 12;
%recessionplot;
set(gcf,'units','points','position',[10,10,800,200])


% Curvature:: 
figure(gcf);
plot(time,xi_smooth_DNS(3,:),'SeriesIndex',1,'LineStyle', '-.', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSL(3,:)*0.5,'SeriesIndex',5,'LineStyle', '--', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSS(3,:)*0.5,'SeriesIndex',7,'LineStyle', ':', 'LineWidth', 2);hold on;
plot(time, empCurv, 'k', 'LineWidth',1); hold on;

plot([min(xlim()),max(xlim())],[0,0], 'k--', 'LineWidth', 0.2);
ax = gca; 
%txt1 = strcat("Ann. average return = ",annRet,"% ", annRetStd);
%txt2 = strcat("Max Drawdown =  ",MaxDD, "%");
%text(time(5),3.30,txt1,'FontSize',13)
%text(time(5),2.95,txt2,'FontSize',13)
ylim([-5, 4]);
xticks(time(1:60:end));
datetick('x', 'yyyy', 'keepticks')
ax.FontSize = 12;
%recessionplot;
set(gcf,'units','points','position',[10,10,800,200])

figure(gcf);
plot(time,xi_smooth_DNS(4,:),'SeriesIndex',1,'LineStyle', '-.', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSL(4,:),'SeriesIndex',5,'LineStyle', '--', 'LineWidth', 2);hold on;
plot(time,xi_smooth_MSS(4,:),'SeriesIndex',7,'LineStyle', ':', 'LineWidth', 2);hold on;
plot(time, empL_inf+1.2, 'k', 'LineWidth',1); hold on;
plot([min(xlim()),max(xlim())],[0,0], 'k--', 'LineWidth', 0.2);
ax = gca; 
%txt1 = strcat("Ann. average return = ",annRet,"% ", annRetStd);
%txt2 = strcat("Max Drawdown =  ",MaxDD, "%");
%text(time(5),3.30,txt1,'FontSize',13)
%text(time(5),2.95,txt2,'FontSize',13)
ylim([-2, 5]);
xticks(time(1:60:end));
datetick('x', 'yyyy', 'keepticks')
ax.FontSize = 12;
%recessionplot;
set(gcf,'units','points','position',[10,10,800,200])