function [pr_sm00,pr_sm01 , pr_sm10, pr_sm11,pr_sm0,pr_sm1] = KimSmoother(pr_upd0, pr_pred0, S0_pr, S1_pr)
%pr_upd0 = pr_upd
%pr_pred0 = pr_pred0

%PROC SMOOTH(pr_upd0,pr_pred0);

%         @pr_upd0 contains Pr[S_t|Y_t]@
%         @pr_pred0 contains Pr[S_t|Y_{t-1}]@

%S1_pr  @Pr[St=1/St-1=1]@
%S0_pr  @Pr[St=0/St-1=0]@

pr_upd1=1-pr_upd0;
pr_pred1=1-pr_pred0;


T = 252; % data size. 

%j_iter=upd-1;
%do until j_iter < 1;

pr_sm00 = zeros(T,1);
pr_sm01 = zeros(T,1);
pr_sm10 = zeros(T,1);
pr_sm11 = zeros(T,1);

pr_sm0 = zeros(T,1);
pr_sm1 = zeros(T,1);

pr_sm0(T)=pr_upd0(T);     % pr_sm0 will contain Pr[S_t|Y_T]@
pr_sm1(T)=pr_upd1(T);
% pr_sm0=pr_upd0;     % pr_sm0 will contain Pr[S_t|Y_T]@
% pr_sm1=pr_upd1; 

for j_iter = T-1:-1:1
   %The followings are P[S_t, S_t+1|Y_T] @

   pr_sm00(j_iter) =pr_sm0(j_iter+1)*S0_pr*    pr_upd0(j_iter)/ pr_pred0(j_iter+1);

   pr_sm01(j_iter) =pr_sm1(j_iter+1)*(1-S0_pr)*pr_upd0(j_iter)/ pr_pred1(j_iter+1);

   pr_sm10(j_iter) =pr_sm0(j_iter+1)*(1-S1_pr)*pr_upd1(j_iter)/ pr_pred0(j_iter+1);

   pr_sm11(j_iter) =pr_sm1(j_iter+1)*S1_pr*    pr_upd1(j_iter)/ pr_pred1(j_iter+1);

   pr_sm0(j_iter)=pr_sm00(j_iter)+pr_sm01(j_iter);
   pr_sm1(j_iter)=pr_sm10(j_iter)+pr_sm11(j_iter);
end

%RETP(pr_sm0); @This proc returns Pr[S_t=0|Y_T]@
end