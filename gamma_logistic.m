% Iterative Weighted Logistic Algorithm (IWLA) 
% more stable (but slower) than Newton method!

% ----- INPUT -----
% y: n by 1 response vector
% x: n by p covariate matrix
% gamma: value of gamma
% lam: penalty for stablization
% b1: initial value of beta
%
% ----- OUTPUT -----
% b1: coefficient estimate
% wi: weight for each subject   
% cov_est: estimated asymptotic covariance matrix

function [b1, wi, cov_est] = gamma_logistic(y, x, gamma, lam, b1)

if nargin < 4
    lam = 0.001;
    b1 = mis_logistic(y, x, 0.05, 1*lam);
elseif nargin < 5
    b1 = mis_logistic(y, x, 0.05, 1*lam);
end
    
[n,p]=size(x);
p=p-1;
DI=lam*eye(p+1);
% DI(1,1)=0;

d=1; sg=1;  
while (d>10^(-3) && sg <= 100)
    b0=b1;
    ei_gamma = exp(x*b0*(gamma+1)); 
    wi = ( (ei_gamma.^y)./(1+ei_gamma) ).^(gamma/(gamma+1));     
    [b1] = logi_fit(y, x, lam/(gamma+1), wi);
    b1 = b1./(gamma+1);
    
    d=norm(b1-b0)/norm(b0);
    sg=sg+1;
end
if sg>100
    display(['Not Convergne at gamma = ', num2str(gamma)])
%     [b0, b1]
%     d
end

%% asymptotic covariancve
ei_gamma = exp((gamma+1).*(x*b1)); 
wi = ( (ei_gamma.^y)./(1+ei_gamma) ).^(gamma/(gamma+1)); 
pi = ei_gamma./(1+ei_gamma);
vi = pi.*(1-pi);
fi = ((1+ei_gamma)./(1+exp(x*b1)).^(gamma+1)).^(1/(gamma+1));
delta = gamma*(x'*diag(wi.*(vi-(y-pi).^2) ./ n)*x);

U1 = (x'*diag((wi.*(y-pi)).^2 ./ n)*x);   % better 
U2 =  (x'*diag(vi.^((2*gamma+1)/(gamma+1)) ./ n)*x);

H1 = (x'*diag(wi.*vi ./ n)*x) + delta + DI;
H2 = (x'*diag(fi.*vi ./ n)*x) + delta + DI;   % better

% cov1 = H1\U1/H1 / n;
cov_est = H2\U1/H2 / n;   % best
% cov3 = H2\U2/H2 / n;   % better

% Q = eye(p+1) - b1*b1'./norm(b1)^2;
% IF = diag(wi.*(y-pi))*x/H1*Q./norm(b1) ;
% IF = mean(sum(IF.^2, 2));
end



function [b1, wi] = mis_logistic(y, x, eta, lam)

if nargin < 4
    lam = 0.001;
end
    
[n,p] = size(x);
p = p-1;
DI = (lam)*eye(p+1);
% DI(1,1) = 0;

% b1=logi_fit(y, x, lam);
b1=pinv(x'*x+10*DI)*(x'*y); % initial value
val1=obj(y, x, eta, lam, b1);
d=1; sg=1; itr=100;
while d > 10^(-3) && sg <= itr
    b0=b1;
    val0=val1;
    wi = (1-2*eta)./(1-eta+eta*exp(x*b0))./(1-eta+eta*exp(-x*b0));    
    pi = (1-eta)*(1+exp(-x*b0)).^(-1) + eta*(1+exp(x*b0)).^(-1);
    vi = (1-2*eta).*(exp(x*b0)./(1+exp(x*b0)).^2);   
    
    HG = (x'*diag(wi.*vi./n)*x+DI)\(x'*diag(wi./n)*(y-pi)-DI*b0);
    [val1, b1] = amj(y, x, eta, lam, val0, b0, HG);
        
    d=norm(b1-b0)/norm(b0);
    sg=sg+1;
end
if sg > itr
    display(['Not converge at eta = ', num2str(eta)])
end
end

%%% Armijo %%%
function [val1, b1] = amj(y, x, eta, lam, val0, b0, HG)
    rate = 1;
    amj=1;   
    amj_num = 30;
    
    b1 = b0 + rate * HG;
    val1 = obj(y, x, eta, lam, b1);
    while val1 < val0 - 10^-8 && amj <= amj_num;
        b1 = b0 + (0.5^amj * rate) * HG;
        val1 = obj(y, x, eta, lam, b1);
        amj = amj + 1;
    end
    if amj > amj_num
        display('mis-logi: armijo limit')
    end
end


function val = obj(y, x, eta, lam, beta)
    pi = (1-eta)*(1+exp(-x*beta)).^(-1) + eta*(1+exp(x*beta)).^(-1);
    val = mean(y.*log(pi)+(1-y).*log(1-pi)) - 0.5*lam*norm(beta)^2; 
end

















