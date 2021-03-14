function [output, LL, xi_0, xi_1, P_0, P_1, xi00_pred, xi01_pred, xi10_pred, xi11_pred, RMSE, AIC, BIC, xi_smooth, pr_sm0] = NegativeLogLikeMS(param_vector, y252)

%==========================================================================
%      As in ATSE:   
%      y_t =     H_t * xi_t +  w_t ,  w_t ~ N(0,R)
%      xi_t+1 =  F   * xi_t +  v_t.   v_t ~ N(0,Q) 
%==========================================================================

ft     = param_vector(1:4);
lambda = param_vector(5);
alpha  = param_vector(6);
S0_Pr  = param_vector(7);                  % Pr[St=0|St-1=0]
S1_Pr  = param_vector(8);                  % Pr[St=1|St-1=1]
sigmas = param_vector(9:12);               % 
mu_xi = param_vector(13:16);

[F, H, R, Q, dt, ct] = DNS(ft, mu_xi, lambda, alpha, sigmas);

T = size(y252,1);
mat = [1, 2, 5, 10, 20, 30];

% Initializing the filter:
% xi_prev0 = (eye(4) - F) \ mu_xi';  %inv(eye(4) - F) * mu_state0; % Initialize at the unconditional mean
% xi_prev1 = xi_prev0; %(eye(4) - F) \ mu_xi';  %inv(eye(4) - F) * mu_state1;

xi_prev0 = zeros(4,1); % work with demeaned states, so better to initialize them at 0.
xi_prev1 = xi_prev0;
% 
% xi_prev0 = mu_xi';
% xi_prev1 = mu_xi';

% Starting values are important, this gives unconditional value.
vecp_prev = (eye(16) - kron(F,F)) \ reshape(Q,[],1); % Initialize at the unconditional covariance matrix
%diffusion
vecp_prev = eye(16) * 10^(16); 
P_prev0 = [vecp_prev(1:4,1) vecp_prev(5:8,1) vecp_prev(9:12,1) vecp_prev(13:16,1) ];
P_prev1 = P_prev0;

% initial probabilities t=1
Prob1 = (1 - S0_Pr) / (2 - S1_Pr - S0_Pr); % Pr[S_0=1|Y0], Steady state prob.
Prob0 = 1 - Prob1;                         % Pr[S_0=0|Y0], Steady state prob.

LL = zeros(1,T);
f_error = zeros(T,12);

xi00_pred = zeros(4,T);
xi01_pred = zeros(4,T);
xi10_pred = zeros(4,T);
xi11_pred = zeros(4,T);

P00_pred = zeros(4,4,T);
P01_pred = zeros(4,4,T);
P10_pred = zeros(4,4,T);
P11_pred = zeros(4,4,T);
%P_pred = zeros(4,4,T+1);

xi_0 = zeros(4,T);
xi_1 = zeros(4,T);
P_0 = zeros(4,4,T);
P_1 = zeros(4,4,T);


PR_TT0M=zeros(T,1);  % stores Pr[S_t=0|Y_t]@ necessary for smoothing
PR_TL0M=zeros(T,1);  % Pr[S_t=0|Y_{t-1}]

%pred = TL : t | t -1
%prev = LL : t-1 | t-1
%upd = TT : t | t  

% Start iteration
for it = 1:T
    PR_TL0M(it)= S0_Pr*Prob0 + (1-S1_Pr)*Prob1;%     @Pr[St=0/Yt-1]@
    
    %  @===================================================================@
    %  @=======================Kalman Filter===============================@
    %  @===================================================================@
    
    %  @===================PREDICTION=============================@
    xi_pred00 = dt + F * xi_prev0;  % When S_{t-1}=0, S_{t}=0 -> xi_{t|t-1}^{0,0}
    xi_pred01 = dt + F * xi_prev0;  
    xi_pred10 = dt + F * xi_prev1; 
    xi_pred11 = dt + F * xi_prev1;  
    
    %[4x4] Xi conditional variance
    P_pred00 = F * P_prev0 * F' + Q;            % P_{t|t-1}^{0,0}
    P_pred01 = F * P_prev0 * F' + Q; 
    P_pred10 = F * P_prev1 * F' + Q; 
    P_pred11 = F * P_prev1 * F' + Q; 
   
    %[12x1]     
    forcast_error00= y252(it,:)'- H * xi_pred00-ct;     % Conditional forecast error: eta_{t|t-1}^{0,0}
    forcast_error01= y252(it,:)'- H * xi_pred01-ct;
    forcast_error10= y252(it,:)'- H * xi_pred10-ct;
    forcast_error11= y252(it,:)'- H * xi_pred11-ct;
    
    %[12x12]
    SS00= H * P_pred00 * H' +R;               % Conditional variance of forecast error: f_{t|t-1}^({0,0}
    SS01= H * P_pred01 * H' +R;
    SS10= H * P_pred10 * H' +R;
    SS11= H * P_pred11 * H' +R;
    
    %@=======================UPDATING===============================@
    % [4x1]  
    xi_upd00 = xi_pred00 + (P_pred00 * H') * (SS00 \ forcast_error00); % xi_{t|t}^{0,0}
    xi_upd01 = xi_pred01 + (P_pred01 * H') * (SS01 \ forcast_error01);
    xi_upd10 = xi_pred10 + (P_pred10 * H') * (SS10 \ forcast_error10);
    xi_upd11 = xi_pred11 + (P_pred11 * H') * (SS11 \ forcast_error11);
    
    % [4x4]
    P_upd00 = (eye(4) - (P_pred00 * H') * (SS00 \ H )) * P_pred00; % P_{t|t}^{0,0}
    P_upd01 = (eye(4) - (P_pred01 * H') * (SS01 \ H )) * P_pred01;
    P_upd10 = (eye(4) - (P_pred10 * H') * (SS10 \ H )) * P_pred10;
    P_upd11 = (eye(4) - (P_pred11 * H') * (SS11 \ H )) * P_pred11;
    
    %@===================================================================@
    %@======================== Hamilton Filter ==========================@
    %@===================================================================@
    % [1x1] V_prob computes likelihood according to multivariate normal distribution.
    Like_norm00= V_Prob(forcast_error00,SS00)*    S0_Pr*Prob0;    % Pr[St,Yt|Yt-1]
    Like_norm01= V_Prob(forcast_error01,SS01)*(1-S0_Pr)*Prob0;
    Like_norm10= V_Prob(forcast_error10,SS10)*(1-S1_Pr)*Prob1;
    Like_norm11= V_Prob(forcast_error11,SS11)*    S1_Pr*Prob1;
    
    % [1x1] CONDITIONAL DENSITY TIMES WEIGHT
    Cond_density=Like_norm00+Like_norm01+Like_norm10+Like_norm11; % eq 3 flowchart, 
    
    % [1x1] WEIGHTED AVERAGE OF CONDITIONAL DENSITIES: f(y_t|Y_{t-1})
    PRO_00=Like_norm00/Cond_density;       % Pr[St,St-1|Yt], 4th equation p.105 Flowchart
    PRO_01=Like_norm01/Cond_density;
    PRO_10=Like_norm10/Cond_density;
    PRO_11=Like_norm11/Cond_density;
    
    
    %SMOOTH 1
    %B_TT(:,it)=xi_upd00*PRO_00+xi_upd01*PRO_01+xi_upd10*PRO_10+xi_upd11*PRO_11;
    %DCCI_M(it,1)=B_TT(1,1);
    
    %[1x1]
    Prob0=PRO_00+PRO_10;       % Pr[St=0|Yt], 5th equation p.105 Flowchart
    Prob1=PRO_01+PRO_11;       % Pr[St=1|Yt]
    
    %SMOOTH 2
    PR_TT0M(it)=Prob0;
    
    %@===================================================================@
    %@======================= Collapsing Terms ==========================@
    %@===================================================================@
    % Idea: from the 4 possible transitions, output only 2: state 0 or 1.
    %[4x1]
    xi_prev0=(PRO_00*xi_upd00 + PRO_10*xi_upd10)/Prob0;   % xi_{t|t}^{0}
    xi_prev1=(PRO_01*xi_upd01 + PRO_11*xi_upd11)/Prob1;   % xi_{t|t}^{1}
    
    %[4x4] 
    P_prev0=(PRO_00*(P_upd00+(xi_prev0-xi_upd00)*(xi_prev0-xi_upd00)')+ ... % P_{t|t}^{0}
        PRO_10*(P_upd10+(xi_prev0-xi_upd10)*(xi_prev0-xi_upd10)'))/Prob0;
    %[4x4]
    P_prev1=(PRO_01*(P_upd01+(xi_prev1-xi_upd01)*(xi_prev1-xi_upd01)')+ ... % P_{t|t}^{1}
        PRO_11*(P_upd11+(xi_prev1-xi_upd11)*(xi_prev1-xi_upd11)'))/Prob1;
    
    LL(it) = -log(Cond_density); % Negative log like!!!! 
    xi_0(:,it) = xi_prev0;
    xi_1(:,it) = xi_prev1;
    P_0(:,:,it) = P_prev0;
    P_1(:,:,it) = P_prev1;
    
    xi00_pred(:,it) = dt + F * xi_prev0;
    xi01_pred(:,it) = dt + F * xi_prev0;
    xi10_pred(:,it) = dt + F * xi_prev1;
    xi11_pred(:,it) = dt + F * xi_prev1;
    
    P00_pred(:,:,it) = F * P_prev0 * F' + Q;            % P_{t|t-1}^{0,0}
    P01_pred(:,:,it) = F * P_prev0 * F' + Q; 
    P10_pred(:,:,it) = F * P_prev1 * F' + Q; 
    P11_pred(:,:,it) = F * P_prev1 * F' + Q; 
    
    
    f_error(it,:) = (forcast_error00 + forcast_error01+ forcast_error10 + forcast_error11)/4;
end

% Sum over all observations (you may use a `burning period' of a few observations - but check if it matters)
output = sum( LL(20:end) );

RMSE= sqrt(mean(   ((f_error(20:end,:)).^2),1) );
[AIC,BIC] = aicbic(-output, length(param_vector));
output


%% Smoothing!

[pr_sm00,pr_sm01 , pr_sm10, pr_sm11,pr_sm0,pr_sm1]= KimSmoother(PR_TT0M,PR_TL0M, S0_Pr, S1_Pr);% @Pr[S_t=0|Y_T], Smoothed probabilities@


xi00_smooth = zeros(4,T);
xi01_smooth = zeros(4,T);
xi10_smooth = zeros(4,T);
xi11_smooth = zeros(4,T);

xi0_smooth = zeros(4,T);
xi1_smooth = zeros(4,T);

xi_smooth = zeros(4,T);
P00_smooth = zeros(4,4,T);
P01_smooth = zeros(4,4,T);
P10_smooth = zeros(4,4,T);
P11_smooth = zeros(4,4,T);

for it = T-1:-1:1
    
   P_tilde00 = P_0(:,:,it) * (F\P00_pred(:,:,it));
   P_tilde01 = P_0(:,:,it) * (F\P01_pred(:,:,it)); 
   P_tilde10 = P_1(:,:,it) * (F\P10_pred(:,:,it));
   P_tilde11 = P_1(:,:,it) * (F\P11_pred(:,:,it));
   
   xi00_smooth(:,it) = xi_0(:,it) + P_tilde00 * (xi00_smooth(:,it+1) -  xi00_pred(:,it));
   xi01_smooth(:,it) = xi_0(:,it) + P_tilde01 * (xi01_smooth(:,it+1) -  xi01_pred(:,it));
   xi10_smooth(:,it) = xi_1(:,it) + P_tilde10 * (xi10_smooth(:,it+1) -  xi10_pred(:,it));
   xi11_smooth(:,it) = xi_1(:,it) + P_tilde11 * (xi11_smooth(:,it+1) -  xi11_pred(:,it));
   
   % not needed for our plots!
%    P00_smooth(:,:,it) = P_0(:,:,it) + P_tilde00*(P00_smooth(:,:,it+1) - P00_pred(:,:,it))* P_tilde00';
%    P01_smooth(:,:,it) = P_0(:,:,it) + P_tilde00*(P00_smooth(:,:,it+1) - P00_pred(:,:,it))* P_tilde00';
%    P10_smooth(:,:,it) = P_0(:,:,it) + P_tilde00*(P00_smooth(:,:,it+1) - P00_pred(:,:,it))* P_tilde00';
%    P11_smooth(:,:,it) = P_0(:,:,it) + P_tilde00*(P00_smooth(:,:,it+1) - P00_pred(:,:,it))* P_tilde00';

   xi0_smooth(:,it) =  (xi00_smooth(:,it)*pr_sm00(it)  +xi10_smooth(:,it) *pr_sm10(it))/pr_sm0(it);
   xi1_smooth(:,it) =  (xi01_smooth(:,it)*pr_sm01(it)  +xi11_smooth(:,it) *pr_sm11(it))/pr_sm1(it);
   
   % Final smoothing!
   %xi_smooth(:,it) =  xi0_smooth(:,it)* pr_sm0(it) + xi1_smooth(:,it) * pr_sm1(it);
   xi_smooth(:,it) = (xi00_smooth(:,it)*pr_sm00(it)  +xi10_smooth(:,it) *pr_sm10(it)) +(xi01_smooth(:,it)*pr_sm01(it)  +xi11_smooth(:,it) *pr_sm11(it));
end

%xi_smooth(:,it) =  xi0_smooth(:,it)* pr_sm0(it) + xi1_smooth(:,it) * pr_sm1(it);

end