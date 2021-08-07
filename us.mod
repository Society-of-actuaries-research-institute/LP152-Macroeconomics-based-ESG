% open economy DSGE model
% by Kailan Shang
% Created on March 24, 2019
% Last version; Dec. 12, 2019

%===================================================================================================
% Cleaning
%===================================================================================================
close all;

%===================================================================================================
% Declaration of endogenous variables
%===================================================================================================
var
%---------------------------------------------------------------------------------------------------
% Aggregate variables
%---------------------------------------------------------------------------------------------------
y                       $\hat{y}$               % Output    
c                       $\hat{c}$               % Consumption
i                       $\hat{i}$               % Investment
g                       $\hat{g}$               % Government spending
imp                     $\hat{imp}$             % Imports
ex                      $\hat{exp}$             % Exports
q                       $\hat{q}$               % Cash holding

c_m                     $\hat{c}^m$             % Imported consumer goods
i_m                     $\hat{i}^m$             % Imported investment goods
i_d                     $\hat{i}^d$             % Domestic investment goods

%---------------------------------------------------------------------------------------------------
% Price indices
%---------------------------------------------------------------------------------------------------
pi_cbar                 $\hat{\bar{\pi}}^c$     % Inflation target
pi_c                    $\hat{\pi}^c$           % CPI inflation
pi_i                    $\hat{\pi}^{i}$         % Investment price inflation
pi_d                    $\hat{\pi}^{d}$         % Domestic goods inflation 
pi_mc                   $\hat{\pi}^{m,c}$       % Imported consumption good inflation
pi_mi                   $\hat{\pi}^{m,i}$       % Imported investment good inflation
pi_x                    $\hat{\pi}^{x}$         % Export good inflation

%---------------------------------------------------------------------------------------------------
% Marginal cost
%---------------------------------------------------------------------------------------------------
mc_d                    $\hat{mc}^{d}$          % Marginal cost: domestic good
mc_mc                   $\hat{mc}^{m,c}$        % Marginal cost: imported consumption good
mc_mi                   $\hat{mc}^{m,i}$        % Marginal cost: imported investment good
mc_x                    $\hat{mc}^{x}$          % Marginal cost: export good

%---------------------------------------------------------------------------------------------------
% Input Cost
%---------------------------------------------------------------------------------------------------
r_k                     $\hat{r}^k$             % Rental rate of capital
w                       $\hat{w}$               % Real wage

%---------------------------------------------------------------------------------------------------
% Capital
%---------------------------------------------------------------------------------------------------
P_k                     $\hat{P}^k$             % Price of capital
mu_z                    $\hat{\mu}^z$           % Trend productivity
kbar                    $\hat{k}$               % Capital stock
k                       $\hat{k}^s$             % Capital services
u                       $\hat{u}$               % Capital utilisation

%---------------------------------------------------------------------------------------------------
% Exchange rate and interest rate
%---------------------------------------------------------------------------------------------------
dS                      $\Delta\hat{S}$         % Nominal depreciation
x                       $\hat{x}$               % Real exchange rate
R                       $\hat{R}$               % Policy interest rate (repo)
a                       $\hat{a}$               % Net foreign assets
psi_z                   $\hat{\psi}^z$          % Optimal asset holdings

%---------------------------------------------------------------------------------------------------
% Labour market
%---------------------------------------------------------------------------------------------------
H                       $\hat{H}$               % Labor supply
E                       $\hat{E}$               % Employment

%---------------------------------------------------------------------------------------------------
% Relative prices (6)
%---------------------------------------------------------------------------------------------------
gamma_mcd               $\hat{\gamma}^{m,c,d}$ 
gamma_mid               $\hat{\gamma}^{m,i,d}$
gamma_xstar             $\hat{\gamma}^{x,*}$
gamma_cd                $\hat{\gamma}^{c,d}$
gamma_id                $\hat{\gamma}^{i,d}$
gamma_f                 $\hat{\gamma}^{f}$

%---------------------------------------------------------------------------------------------------
% Foreign economy (3)
%---------------------------------------------------------------------------------------------------
R_star                  $\hat{R}^{*}$
pi_star                 $\hat{\pi}^{*}$
y_star                  $\hat{y}^{*}$
e_ystar
e_pistar

%--------------------------------------------------------------------------------------------------- 
% AR(1) shock processes (11)
%---------------------------------------------------------------------------------------------------
e_c                     $\hat{\xi}^{c}$
e_i                     $\hat{\xi}^{i}$
e_a                     $\hat{\tilde{\phi}}^{}$
e_z                     $\hat{\varepsilon}^{}$
e_H                     $\hat{\xi}^{H}$
lambda_x                $\hat{\lambda}^{x}$
lambda_d                $\hat{\lambda}^d$
lambda_mc               $\hat{\lambda}^{m,c}$
lambda_mi               $\hat{\lambda}^{m,i}$
z_tildestar             $\hat{\tilde{z}}^{*}$

%---------------------------------------------------------------------------------------------------
% Tax rates
%---------------------------------------------------------------------------------------------------
i_k                     $\hat{\iota}^{k}$         %aggregate capital gain tax rate
i_c                     $\hat{\iota}^{c}$         %aggregate consumption tax rate
i_y                     $\hat{\iota}^{y}$         %aggregate income tax rate
i_w                     $\hat{\iota}^{w}$         %aggregate payroll tax rate


%===================================================================================================
% Observable variables
%===================================================================================================
R_                      %domestic policy interest rate
pi_c_					%CPI
pi_cbar_                %inflation target
pi_i_                   %investment inflation
pi_d_                   %domestic goods inflation
dy_                     %change in real GDP
dc_                     %change in consumption
di_                     %change in investment
dimp_                   %change in import
dex_                    %change in export
dy_star_                %change in foreign real GDP
pi_star_                %change in foreign inflation rate
R_star_                 %foreign policy interest rate
dE_                     %change in employment
dS_                     %change in nominal exchange rate (rather than real exchange rate)
dw_                     %change in employee compensation

;

%===================================================================================================
% Declaration of exogenous variables
%===================================================================================================

varexo

%---------------------------------------------------------------------------------------------------
% Shock terms
%---------------------------------------------------------------------------------------------------

eps_c                       $\varepsilon^{c}$
eps_i                       $\varepsilon^{i}$
eps_g                       $\varepsilon^{g}$

eps_r                       $\varepsilon^{r}$     % shock term in central bank policy interest rate decision
eps_a                       $\varepsilon^{a}$     % shock term in e_a
eps_z                       $\varepsilon^{z}$     % shock term in e_z
eps_H                       $\varepsilon^{H}$
eps_pi_cbar                 $\varepsilon^{\bar{\pi}^c}$
eps_x                       $\varepsilon^{x}$
eps_d                       $\varepsilon^{d}$
eps_mc                      $\varepsilon^{m,c}$
eps_mi                      $\varepsilon^{m,i}$
eps_z_tildestar             $\varepsilon^{\tilde{z}^*}$    
eps_mu_z                    $\varepsilon^{\mu^z}$    

eps_Rstar                   $\varepsilon^{R^*}$
eps_pistar                  $\varepsilon^{\pi^*}$
eps_ystar                   $\varepsilon^{y^*}$    

eps_i_k                     $\varepsilon^{{\iota}^k}$
eps_i_y                     $\varepsilon^{{\iota}^y}$


%---------------------------------------------------------------------------------------------------
% Measurement errors
%---------------------------------------------------------------------------------------------------
me_w
me_E
me_pi_d
me_pi_i
me_y
me_c
me_i
me_imp
me_ex
me_ystar
;

%---------------------------------------------------------------------------------------------------
% Model parameters
%---------------------------------------------------------------------------------------------------
parameters

beta           $\beta$     
delta          $\delta$   
alpha          $\alpha$   
A_L            $A_L$    
sigma_L        $\sigma_L$
A_q            $A_q$
sigma_q        $\sigma_q$
sigma_a        $\sigma_a$
b              $b$
phi_a          $\phi_a$
phi_s          $\phi_s$
phi_i          $\phi_i$
theta_w        $\theta_w$
theta_d        $\theta_d$    
theta_mc       $\theta_mc$    
theta_mi       $\theta_mi$    
theta_x        $\theta_x$    
theta_e        $\theta_e$    
xi_w        $\xi_w$    
xi_d        $\xi_d$    
xi_mc       $\xi_mc$
xi_mi       $\xi_mi$
omega_i        $\vartheta_i$
omega_c        $\vartheta_c$
eta_c          $\eta_c$
eta_i          $\eta_i$
eta_f          $\eta_f$
rho_r          $\rho_R$
phi_pi           $\phi_{\pi}$
phi_y            $\phi_{y}$
phi_x            $\phi_{x}$
phi_dpi          $\phi_{\Delta\pi}$
phi_dy           $\phi_{\Delta y}$
phi_pi_star      $\phi_{\pi}^*$
phi_y_star      $\phi_{y}^*$
rho_pi         $\rho_{\pi}$  
mu_b_s        $\mu^m$
gr             $gr$
nu_s          $\nu$
mu_z_s        $\mu^z$
lambda_d_s    $\lambda^d$ 
lambda_mc_s   $\lambda^mc$
lambda_mi_s   $\lambda^mi$
lambda_w_s    $\lambda^w$
gamma_cmc_s   $\gamma^{c,mc}$
gamma_imi_s   $\gamma^{i,mi}$
gamma_cd_s    $\gamma^{c,d}$
gamma_id_s    $\gamma^{i,d}$
gamma_dc_s    $\gamma^{d,c}$
gamma_mcc_s   $\gamma^{mc,c}$
eta_mc         $\eta_mc$
eta_mi         $\eta_mi$
k_s           $k^s$
kbar_s        $k$
r_k_s         $r^k$
R_s           $R$
y_s           $y$
i_s           $i$
c_d_s         $c^d$
c_m_s         $c^m$
i_m_s         $i^m$
i_d_s         $i^d$
y_star_s      $y^*$
pi_star_s     $\pi^*$
R_star_s      $R^*$
dS_s          $\Delta{S}$
w_s           $w$
k_H_s         $\frac{k}{H}$
T1             $T_1$
T2             $T_2$
T3             $T_3$
T4             $T_4$
H_s           $H$
c_s           $c$
psi_z_s       $\psi^z$
pi_s          $\pi$
q_s           $q$
b_w            $b_w$
phi_0          $\phi_0$
phi_1          $\phi_1$
phi_2          $\phi_2$
phi_3          $\phi_3$
phi_4          $\phi_4$
phi_5          $\phi_5$
phi_6          $\phi_6$
phi_7          $\phi_7$
phi_8          $\phi_8$
phi_9          $\phi_9$ 
phi_10         $\phi_10$ 
phi_11         $\phi_11$
imp_s         $IMP$

%---------------------------------------------------------------------------------------------------
% Tax rate constant term value 
%---------------------------------------------------------------------------------------------------

c_i_k          $c_{\iota^{k}}$     %capital gain tax rate constant term
c_i_c          $c_{\iota^{c}}$     %consumption tax rate constant term
c_i_y          $c_{\iota^{y}}$     %income tax rate constant term
c_i_w          $c_{\iota^{w}}$     %payroll tax rate constant term
c_i_g          $c_{\iota^{g}}$     %government spending constant term

%---------------------------------------------------------------------------------------------------
% Tax rate steady value 
%---------------------------------------------------------------------------------------------------

ik_s             $\iota^{k}$         %stable aggregate capital gain tax rate
ic_s             $\iota^{c}$         %stable aggregate consumption tax rate
iy_s             $\iota^{y}$         %stable aggregate income tax rate
iw_s             $\iota^{w}$         %stable aggregate payroll tax rate

%---------------------------------------------------------------------------------------------------
% Tax and fiscal budget parameters
%---------------------------------------------------------------------------------------------------

rho_i_k_k      $\rho_{\i^k_k}$
rho_i_c_c      $\rho_{\i^c_c}$
rho_i_y_y      $\rho_{\i^y_y}$
rho_i_w_w      $\rho_{\i^w_w}$
eps_i_w                      $\varepsilon^{{\iota}^k}$
eps_i_c                      $\varepsilon^{{\iota}^c}$

%---------------------------------------------------------------------------------------------------
% AR(1) coefficient
%---------------------------------------------------------------------------------------------------
rho_mu_z       $\rho_{\mu^z}$
rho_z          $\rho_{z}$
rho_i          $\rho_{i}$
rho_ztildestar $\rho_{\tilde{z}^*}$
rho_g          $\rho_{g}$
rho_nu         $\rho_{\nu}$
rho_c          $\rho_{c}$
rho_H          $\rho_{H}$
rho_a          $\rho_{a}$
rho_lambda_d   $\rho_{\lambda^{d}}$
rho_lambda_mc  $\rho_{\lambda^{m,c}}$
rho_lambda_mi  $\rho_{\lambda^{m,i}}$
rho_lambda_x   $\rho_{\lambda^{x}}$

%---------------------------------------------------------------------------------------------------
% Foreign economy parameters
%---------------------------------------------------------------------------------------------------
rho_Rstar      $\rho_{\R^*}$
sigma_star     $\sigma^*$
xi_star     $\kappa^*$
rho_pistar     $\rho_{\pi^*}$
rho_ystar      $\rho_y^*$
;

%===================================================================================================
% Calibrated parameter values
%===================================================================================================

beta     = 0.9975; % discount factor
delta    = 0.013; % depreciation rate
alpha    = 0.23; % capital share in production
A_L      = 7.5;   % constant in labour disutility function
sigma_L  = 1.00;  % labor supply elasticity
A_q      = 0.436; 
sigma_q  = 10.62;
sigma_a  = 1000000;  % capital utilization cost
b        = 0.708;  % habit formation
phi_a    = 0.252;
phi_s    = 0.5;

%---------------------------------------------------------------------------------------------------
% Investment adjustment cost
%---------------------------------------------------------------------------------------------------

phi_i       = 8.670;

%---------------------------------------------------------------------------------------------------
% Calvo's
%---------------------------------------------------------------------------------------------------

theta_w     = 0.690;
theta_d     = 0.891;
theta_mc    = 0.444;
theta_mi    = 0.721;
theta_x     = 0.612;
theta_e     = 0.787;

%---------------------------------------------------------------------------------------------------
% Inflation indexation
%---------------------------------------------------------------------------------------------------

xi_w     = 0.5;
xi_d     = 0.217;
xi_mc    = 0.220;
xi_mi    = 0.231;

%---------------------------------------------------------------------------------------------------
% Import shares
%---------------------------------------------------------------------------------------------------

omega_i     = 0.48; % imported investment share
omega_c     = 0.36; % imported consumption share

%---------------------------------------------------------------------------------------------------
% Substitution elasticities
%---------------------------------------------------------------------------------------------------

eta_c       = 1.5;  % substitution elasticity consumption
eta_i       = 1.5; % substitution elasticity investment
eta_f       = 1.25;% substitution elasticity foreign

%---------------------------------------------------------------------------------------------------
% Policy rule dy and dpi also have prior distribution. need to change
%---------------------------------------------------------------------------------------------------

rho_r       = 0.948;
phi_pi        = 0.678;
phi_y         = 0.547;
phi_x         = 0;
phi_dpi       = 0.204;
phi_dy        = 0.086;
phi_pi_star   = 4.39;
phi_y_star    = 0.255;

%---------------------------------------------------------------------------------------------------
% Inflation target persistence
%---------------------------------------------------------------------------------------------------

rho_pi   = 0;   % inflation target persistence

%---------------------------------------------------------------------------------------------------
% AR(1) shocks' parameters
%---------------------------------------------------------------------------------------------------

rho_mu_z        = 0.698;
rho_z           = 0.886;
rho_i           = 0.720;
rho_ztildestar  = 0.823829;
rho_c           = 0.892;
rho_H           = 0.676;
rho_a           = 0.955;
rho_lambda_mc   = 0.970;
rho_lambda_mi   = 0.963;
rho_lambda_x    = 0.886;
rho_lambda_d    = 0;
rho_g           = 0.815;
rho_nu          = 0.8;

%---------------------------------------------------------------------------------------------------
%Tax and fiscal budget VAR model parameters
%---------------------------------------------------------------------------------------------------
c_i_k        = 0.3918;
c_i_c        = 0.0651;
c_i_y        = 0.1242;
c_i_w        = 0.0765;
c_i_g        = 0;
rho_i_k_k    = 0;
rho_i_c_c    = 0;
rho_i_y_y    = 0;
rho_i_w_w    = 0;
eps_i_c      = 0;
eps_i_w      = 0;

%---------------------------------------------------------------------------------------------------
% Steady state parameters
%---------------------------------------------------------------------------------------------------
 1.0017   1.0151
gr       = 0.178;   % G/Y share
nu_s    = 0.15;    % share of wage bill financed
mu_z_s  = 1.004;  % technology growth

pi_star_s = 0.9998;

lambda_d_s  = 1.1; 
lambda_mc_s = 1.1; 
lambda_mi_s = 1.1; 
lambda_w_s = 1.05; 

rho_Rstar  = 0.984265;
sigma_star = 1.3045;
xi_star = 0.2;
rho_pistar = 0.8;
rho_ystar  = 0.9;

%---------------------------------------------------------------------------------------------------
% Calculated steady state parameters
%---------------------------------------------------------------------------------------------------

ik_s = c_i_k/(1-rho_i_k_k);
ic_s = c_i_c/(1-rho_i_c_c);
iy_s = c_i_y/(1-rho_i_y_y);
iw_s = c_i_w/(1-rho_i_w_w);
pi_s       = mu_b_s/mu_z_s; %S.1 
R_s        = (pi_s*mu_z_s-beta*ik_s)/(beta*(1-ik_s)); %S.2 
eta_mc      = lambda_mc_s/(lambda_mc_s-1); %elasticity of imported goods %S.3
eta_mi      = lambda_mi_s/(lambda_mi_s-1); %elasticity of imported investment %S.4 
gamma_cd_s = ((1-omega_c)+omega_c*(eta_mc/(eta_mc-1))^(1-eta_c))^(1/(1-eta_c)); %S.5 
gamma_cmc_s = ((1-omega_c)*((eta_mc-1)/eta_mc)^(1-eta_c)+omega_c)^(1/(1-eta_c)); %S.6 
gamma_id_s = ((1-omega_i)+omega_i*(eta_mi/(eta_mi-1))^(1-eta_i))^(1/(1-eta_i)); %S.7 
gamma_imi_s = ((1-omega_i)*((eta_mi-1)/eta_mi)^(1-eta_i)+omega_i)^(1/(1-eta_i)); %S.8 
gamma_dc_s = 1/gamma_cd_s;
gamma_mcc_s = 1/gamma_cmc_s;
r_k_s      = (mu_z_s*gamma_id_s-beta*(1-delta)*gamma_id_s)/beta/(1-ik_s); %S.9 
w_s        = (1-alpha)*lambda_d_s^(-1/(1-alpha))*alpha^(alpha/(1-alpha))*r_k_s^(-alpha/(1-alpha)); %%*R_f_s^(-1); %S.10
k_H_s      = alpha/(1-alpha)*mu_z_s*w_s*r_k_s^(-1); %S.11
T1          = ((mu_z_s-beta*b)/((mu_z_s-b)*(1+ic_s)))*gamma_cd_s^(-1);
T2          = (1-omega_c)*gamma_cd_s^eta_c+omega_c*gamma_cmc_s^eta_c;
T3          = (1-gr)/lambda_d_s*mu_z_s^(-alpha)*k_H_s^alpha - ((1-omega_i)*gamma_id_s^eta_i + omega_i*gamma_imi_s^eta_i)*(1-(1-delta)/mu_z_s)*k_H_s;
T4          = (((1-iy_s)/(1+iw_s)/lambda_w_s*w_s)/A_L)^(1/sigma_L);
H_s        = (T4*T1^(1/sigma_L)*(T3/T2)^(-1/sigma_L))^(sigma_L/(1+sigma_L)); %S.12
c_s        = (T3/T2)*H_s; %S.13
psi_z_s    = 1/c_s*T1; %S.14
k_s        = k_H_s*H_s; % k = k/H * H
kbar_s     = k_s; %full utilization
i_s        = (1-(1-delta)/mu_z_s)*k_s; % 93
c_d_s      = (1-omega_c)*gamma_cd_s^eta_c*c_s;
c_m_s      = omega_c*gamma_cmc_s^eta_c*c_s;
i_d_s      = (1-omega_i)*gamma_id_s^eta_i*i_s;
i_m_s      = omega_i*gamma_imi_s^eta_i*i_s;
imp_s      = c_m_s + i_m_s;
y_s        = 1/lambda_d_s*(mu_z_s)^(-alpha)*(k_H_s)^alpha*H_s; %S.15
q_s        = (A_q/(psi_z_s*(R_s-1)*(1-ik_s)))^(1/sigma_q); %S.16
y_star_s   = omega_c*gamma_cmc_s^eta_c*c_s+omega_i*gamma_imi_s^eta_i*i_s;
R_star_s   = (pi_star_s*mu_z_s)/beta;  % without capital tax
dS_s       = pi_s/pi_star_s;

%---------------------------------------------------------------------------------------------------
% Wage optimization interim parameters
%---------------------------------------------------------------------------------------------------

b_w = (lambda_w_s + lambda_w_s*sigma_L - 1)/((1-beta*theta_w)*(1-theta_w)); 
phi_0 = (lambda_w_s*sigma_L - b_w*(1 + beta*(theta_w^2)));
phi_1 = b_w*theta_w;
phi_2 = b_w*beta*theta_w;
phi_3 = -b_w*theta_w;
phi_4 = b_w*beta*theta_w^2; 
phi_5 = b_w*theta_w*xi_w;
phi_6 = -b_w*beta*theta_w^2*xi_w; 
phi_7 = (1 - lambda_w_s);
phi_8 = -iy_s*(1 - lambda_w_s)/(1-iy_s); 
phi_9 = -iw_s*(1 - lambda_w_s)/(1+iw_s); 
phi_10 = -(1 - lambda_w_s)*sigma_L;
phi_11 = -(1 - lambda_w_s);

%==========================================================================
% Model equations
%==========================================================================

model(linear);

%---------------------------------------------------------------------------------------------------
% Capital services (L.1)
%---------------------------------------------------------------------------------------------------
u = k - kbar(-1); 

%---------------------------------------------------------------------------------------------------
% Output (L.2)
%---------------------------------------------------------------------------------------------------
y = lambda_d_s*(e_z + alpha*(k - mu_z) + (1-alpha)*H);

%---------------------------------------------------------------------------------------------------
% Rental rate of capital (L.3)
%---------------------------------------------------------------------------------------------------

r_k = w + mu_z - k + H;

%---------------------------------------------------------------------------------------------------
% Domestic marginal cost (L.4)
%---------------------------------------------------------------------------------------------------
% mc_d = alpha*r_k + (1-alpha)*(w + R_f) - e_z;
mc_d = alpha*r_k + (1-alpha)*w - e_z;

%---------------------------------------------------------------------------------------------------
% Domestic Phillips curve (L.6, L.5 is included in L.6 and not listed here)
%---------------------------------------------------------------------------------------------------
pi_d = pi_cbar + beta/(1+xi_d*beta)*(pi_d(+1) - rho_pi*pi_cbar) 
     + xi_d/(1+xi_d*beta)*(pi_d(-1) - pi_cbar) 
     - (xi_d*beta*(1-rho_pi)/(1+xi_d*beta))*pi_cbar 
     + (((1-theta_d)*(1-theta_d*beta))/(theta_d*(1+xi_d*beta)))*(mc_d + lambda_d);

%---------------------------------------------------------------------------------------------------
% Exporter marginal cost (L.29) used by mc_mc and mc_mi below.
%---------------------------------------------------------------------------------------------------
mc_x = mc_x(-1) + pi_d - pi_x - dS;

%---------------------------------------------------------------------------------------------------    
% Imported Consumption marginal cost (L.26 & L.28) used by L.7 for imported consumption
%---------------------------------------------------------------------------------------------------
mc_mc = - mc_x - gamma_xstar - gamma_mcd;

%---------------------------------------------------------------------------------------------------
% Imported Consumption Phillips curve (L.7)
%---------------------------------------------------------------------------------------------------
pi_mc = pi_cbar + (beta/(1+xi_mc*beta))*(pi_mc(+1) - rho_pi*pi_cbar) 
      + (xi_mc/(1+xi_mc*beta))*(pi_mc(-1) - pi_cbar) 
      - (xi_mc*beta*(1-rho_pi)/(1+xi_mc*beta))*pi_cbar 
      + (((1-theta_mc)*(1-theta_mc*beta))/(theta_mc*(1+xi_mc*beta)))*(mc_mc + lambda_mc);

%---------------------------------------------------------------------------------------------------
% Imported Investment marginal cost (L.27 & L.28) used by L.7 for imported investment
%---------------------------------------------------------------------------------------------------
mc_mi = - mc_x - gamma_xstar - gamma_mid;

%---------------------------------------------------------------------------------------------------
% Imported Investment Phillips curve (L.7)
%---------------------------------------------------------------------------------------------------
pi_mi = pi_cbar + (beta/(1+xi_mi*beta))*(pi_mi(+1) - rho_pi*pi_cbar) 
      + (xi_mi/(1+xi_mi*beta))*(pi_mi(-1) - pi_cbar) 
      - (xi_mi*beta*(1-rho_pi)/(1+xi_mi*beta))*pi_cbar 
      + (((1-theta_mi)*(1-theta_mi*beta))/(theta_mi*(1+xi_mi*beta)))*(mc_mi + lambda_mi);
     
%---------------------------------------------------------------------------------------------------
% Exporter Phillips curve (L.8)
%---------------------------------------------------------------------------------------------------

pi_x = (beta/(1+beta))*pi_x(+1)
     + (1/(1+beta))*pi_x(-1)
     + (((1-theta_x)*(1-theta_x*beta))/(theta_x*(1+beta)))*(mc_x + lambda_x);
     
%---------------------------------------------------------------------------------------------------
% Exports
%---------------------------------------------------------------------------------------------------

ex = -eta_f*gamma_xstar + y_star + z_tildestar; %30

%---------------------------------------------------------------------------------------------------
% Real wage (L.9)
%---------------------------------------------------------------------------------------------------
w = -(1/phi_0)*(phi_1*w(-1) + phi_2*w(+1) + phi_3*(pi_d - pi_cbar) + phi_4*(pi_d(+1) 
  - rho_pi*pi_cbar) + phi_5*(pi_c(-1) - pi_cbar) + phi_6*(pi_c - rho_pi*pi_cbar) + phi_7*psi_z 
  + phi_8*i_y + phi_9*i_w + phi_10*H + e_H); %+ e_H; % + eta_11*e_H);

%---------------------------------------------------------------------------------------------------
% Capital law-of-motion (L.10)
%---------------------------------------------------------------------------------------------------
kbar = (1-delta)/mu_z_s*kbar(-1) - (1-delta)/mu_z_s*mu_z + (1-(1-delta)*1/mu_z_s)*(i + e_i);

%---------------------------------------------------------------------------------------------------
% Consumption Euler equation (L.11)
%---------------------------------------------------------------------------------------------------
c = ((mu_z_s*b)/(mu_z_s^2+beta*(b^2)))*c(-1) + ((mu_z_s*b*beta)/(mu_z_s^2+beta*(b^2)))*c(+1) 
  - ((mu_z_s*b)/(mu_z_s^2+beta*(b^2)))*(mu_z - beta*mu_z(+1)) 
  - (((mu_z_s-b)*(mu_z_s-beta*b))/(mu_z_s^2+beta*(b^2)))*(psi_z + gamma_cd) 
  + ((mu_z_s-b)/(mu_z_s^2+beta*(b^2)))*(mu_z_s*e_c-b*beta*e_c(+1))
  - (((mu_z_s-b)*(mu_z_s-beta*b))/(mu_z_s^2+beta*(b^2)))*ic_s/(1+ic_s)*i_c; 
  
%---------------------------------------------------------------------------------------------------
% Optimal investment (L.12)
%---------------------------------------------------------------------------------------------------

i = (1/((mu_z_s^2*phi_i)*(1+beta)))*((mu_z_s^2*phi_i)*(i(-1) + beta*i(+1)- mu_z + beta*mu_z(+1)) 
  + P_k - gamma_id + e_i); %+ e_i;

%---------------------------------------------------------------------------------------------------
% Price of capital (L.13)
%---------------------------------------------------------------------------------------------------
P_k = ((beta*(1-delta))/mu_z_s)*P_k(+1) + (psi_z(+1) - psi_z) - mu_z(+1) 
    + ((mu_z_s - beta*(1-delta))/mu_z_s)*r_k(+1)
	-((mu_z_s - beta*(1-delta))/mu_z_s)*ik_s/(1-ik_s)*i_k(+1);

%---------------------------------------------------------------------------------------------------
% Capacity utilization (L.14)
%---------------------------------------------------------------------------------------------------
u = 1/sigma_a*r_k - 1/sigma_a*ik_s/(1-ik_s)*i_k;  %- 1/sigma_a*tau_k_s/(1 - tau_k_s)*tau_k;

%---------------------------------------------------------------------------------------------------
% Optimal cash holdings (L.15)
%---------------------------------------------------------------------------------------------------
q = -1/sigma_q * psi_z + 1/sigma_q*ik_s/(1-ik_s)*i_k - 1/sigma_q*R_s/(R_s-1)*R(-1);

%---------------------------------------------------------------------------------------------------
% Optimal asset holdings (L.16)
%---------------------------------------------------------------------------------------------------
psi_z = psi_z(+1) - mu_z(+1) + R_s*(1-ik_s)/(R_s-ik_s*(R_s-1))*R - ik_s*(R_s-1)/(R_s-ik_s*(R_s-1))*i_k(+1) - pi_d(+1);
      
%---------------------------------------------------------------------------------------------------
% Nominal exchange rate (L.18;  L.17 is included in L.18 & L.16)
%---------------------------------------------------------------------------------------------------
% Modified UIP  (L.18; L.17 is included in L.18 & L.16)
R = (1-phi_s)*dS(+1) - phi_s*dS + R_star - phi_a*a + e_a;

%---------------------------------------------------------------------------------------------------
% 62 Tax VAR
%---------------------------------------------------------------------------------------------------
i_k  = c_i_k+ rho_i_k_k*i_k(-1)+ eps_i_k;
i_c  = c_i_c + eps_i_c;
i_y  = c_i_y + eps_i_y;
i_w  = c_i_w + eps_i_w;

%---------------------------------------------------------------------------------------------------
% Goverment spending (62 VAR)
%---------------------------------------------------------------------------------------------------
g = rho_g * g(-1) + eps_g;

%---------------------------------------------------------------------------------------------------
% Policy rule (63 removed four period average)
%---------------------------------------------------------------------------------------------------
R = rho_r*R(-1) + (1-rho_r)*(pi_cbar + phi_pi*(pi_c(-1)-pi_cbar) + phi_y*y(-1) + phi_x*x(-1) 
	+ phi_dpi*(pi_c - pi_c(-1)) + phi_dy*(y - y(-1))) + eps_r;

%---------------------------------------------------------------------------------------------------
% CPI inflation (64)
%---------------------------------------------------------------------------------------------------
pi_c = ((1-omega_c)*(1/gamma_cd_s)^(1-eta_c))*pi_d + ((omega_c)*(1/gamma_cmc_s)^(1-eta_c))*pi_mc;

%---------------------------------------------------------------------------------------------------
% Investment price deflator (64)
%---------------------------------------------------------------------------------------------------
pi_i = ((1-omega_i)*(1/gamma_id_s)^(1-eta_i))*pi_d + ((omega_i)*(1/gamma_imi_s)^(1-eta_i))*pi_mi;

%---------------------------------------------------------------------------------------------------
% Aggregate resource constraint (L.19)
%---------------------------------------------------------------------------------------------------
y = (1-omega_c)*gamma_cd_s^eta_c*c_s/y_s*(c+eta_c*gamma_cd) 
  + (1-omega_i)*gamma_id_s^eta_i*i_s/y_s*(i+eta_i*gamma_id) 
  + gr*g + y_star_s/y_s*(y_star-eta_f*gamma_xstar+z_tildestar) 
  + (1-ik_s)*r_k_s*kbar_s/(y_s*mu_z_s)*(k-kbar(-1));

%---------------------------------------------------------------------------------------------------
% Law of motion for Net Foreign Assets (L.20 (pi_star_s)*(1/beta)==R_s/mu_z_s)
%---------------------------------------------------------------------------------------------------
a = -y_star_s*mc_x - eta_f*y_star_s*gamma_xstar + y_star_s*y_star + y_star_s*z_tildestar 
  + (c_m_s + i_m_s)*gamma_f - (c_m_s*(-eta_c*(1-omega_c)*(gamma_cd_s)^(-(1-eta_c))*gamma_mcd + c) 
  + i_m_s*(-eta_i*(1-omega_i)*(gamma_id_s)^(-(1-eta_i))*gamma_mid + i)) 
  + (pi_star_s/pi_s)*(1/mu_z_s)*a(-1);

%---------------------------------------------------------------------------------------------------
% Relative prices
%---------------------------------------------------------------------------------------------------
gamma_mcd    = gamma_mcd(-1) + pi_mc - pi_d; %L.23
gamma_mid    = gamma_mid(-1) + pi_mi - pi_d; %L.24
gamma_xstar  = gamma_xstar(-1) + pi_x - pi_star; %L.25
gamma_cd     = (1-(1-omega_c)*(gamma_cd_s)^(eta_c-1))*gamma_mcd; %L.21 (a transformed version)
gamma_id     = (1-(1-omega_i)*(gamma_id_s)^(eta_i-1))*gamma_mid; %L.22 (a transformed version)
%gamma_cd     = gamma_cd(-1) + pi_c - pi_d; %L.21 
%gamma_id     = gamma_id(-1) + pi_i - pi_d; %L.22 
gamma_f      = mc_x + gamma_xstar; %L.28 

%---------------------------------------------------------------------------------------------------
% Real exchange rate (L.30 )
%---------------------------------------------------------------------------------------------------
x = -omega_c*(gamma_cmc_s)^(1-eta_c)*gamma_mcd - gamma_xstar - mc_x;

%---------------------------------------------------------------------------------------------------
% Foreign economy 
%---------------------------------------------------------------------------------------------------
R_star = rho_Rstar*R_star(-1) +(1-rho_Rstar)*(phi_pi_star*pi_star + phi_y_star*y_star) + eps_Rstar; %85 
y_star = y_star(+1) - (1/sigma_star)*(R_star - pi_star(+1) + e_ystar); %83
pi_star = beta*pi_star(+1) + xi_star*y_star + e_pistar; %84
e_pistar = rho_pistar*e_pistar(-1) + eps_pistar; %exogenous
e_ystar = rho_ystar*e_ystar(-1) + eps_ystar; %exogenous

%---------------------------------------------------------------------------------------------------
% Imports (86-88)
%---------------------------------------------------------------------------------------------------
c_m = eta_c*(gamma_cd - gamma_mcd) + c;
i_m = eta_i*(gamma_id - gamma_mid) + i;
imp = (c_m_s/imp_s)*c_m + (i_m_s/imp_s)*i_m;
i_d = eta_i*(gamma_mid) + i;


%---------------------------------------------------------------------------------------------------
% Measurement equations
%---------------------------------------------------------------------------------------------------
R_      = R + 100*log(R_s)-1.559; %L.45
pi_c_   = pi_c + 100*log(pi_s) - 0.259; %L.39
pi_cbar_= pi_cbar + 100*log(pi_s) - 0.367; %L.43
pi_i_   = pi_i + 100*log(pi_s) + me_pi_i -0.818; %L.41
pi_d_   = pi_d + 100*log(pi_s) + me_pi_d - 0.342; %L.40
dy_     = y - y(-1) + mu_z + log(mu_z_s)*100 + me_y  - alpha*(k - k(-1) - kbar(-1) + kbar(-2)) + 0.057; %L.31 removing the installed capital that is not utilized.
dc_     = (eta_c/(c_m_s+c_d_s))*(c_d_s*omega_c - c_m_s*(1-omega_c))*(pi_mc 
        - pi_d) + c - c(-1) + mu_z + log(mu_z_s)*100 + me_c + 0.692; %L.32
di_     = (eta_i/(i_m_s+i_d_s))*(i_d_s*omega_i - i_m_s*(1-omega_i))*(pi_mi 
        - pi_d) + i - i(-1) + mu_z + log(mu_z_s)*100 + me_i + 0.869; %L.33
dimp_   = (c_m_s/(c_m_s+i_m_s))*(-eta_c*(1-omega_c)*((gamma_cd_s)^(eta_c-1))*(pi_mc - pi_d) 
        + c - c(-1)) + (i_m_s/(c_m_s+i_m_s))*(-eta_i*(1-omega_i)*((gamma_id_s)^(eta_i-1))*(pi_mi 
        - pi_d) + i - i(-1)) + mu_z + log(mu_z_s)*100 + me_imp + 1.149; %L.35
dex_    = -eta_f*(pi_x - pi_star) + y_star - y_star(-1) + z_tildestar - z_tildestar(-1) + mu_z 
        + log(mu_z_s)*100 + me_ex + 0.793; %L.34
dS_     = dS + 100*log(dS_s) - 0.633; %L.44
dy_star_= y_star - y_star(-1) + z_tildestar - z_tildestar(-1) + mu_z + log(mu_z_s)*100 + me_ystar +0.74; %L.36
pi_star_= pi_star + 100*log(pi_star_s) + 0.025; %L.42
R_star_ = R_star + 100*log(R_star_s) + 0.803; %L.46
dE_      = E - E(-1) + log(1.000682)*100 + me_E - 0.04; %L.38
dw_     = w - w(-1) + mu_z + log(mu_z_s)*100 + pi_d + log(pi_s)*100 + me_w - 0.241; %L.37

%---------------------------------------------------------------------------------------------------
% Employment equation
%---------------------------------------------------------------------------------------------------
E = beta/(1+beta)*E(+1) + (1/(1+beta))*E(-1) 
  + ((1-theta_e)*(1-beta*theta_e))/((1+beta)*theta_e)*(H-E);

%---------------------------------------------------------------------------------------------------
% Shock processes (17) (exogenous)
%---------------------------------------------------------------------------------------------------
e_c         = rho_c*e_c(-1) + eps_c;
e_i         = rho_i*e_i(-1) + eps_i;
e_a         = rho_a*e_a(-1) + eps_a;
e_z         = rho_z*e_z(-1) + eps_z;
e_H         = rho_H*e_H(-1) + eps_H;
lambda_x    = rho_lambda_x*lambda_x(-1) + eps_x;
lambda_d    = rho_lambda_d*lambda_d(-1) + eps_d;
lambda_mc   = rho_lambda_mc*lambda_mc(-1) + eps_mc;
lambda_mi   = rho_lambda_mi*lambda_mi(-1) + eps_mi;
z_tildestar = rho_ztildestar*z_tildestar(-1) + eps_z_tildestar;
mu_z        = rho_mu_z*mu_z(-1) + eps_mu_z;
pi_cbar = rho_pi*pi_cbar(-1) + eps_pi_cbar;

%---------------------------------------------------------------------------------------------------
end;
%---------------------------------------------------------------------------------------------------

%==========================================================================
% Shock distribution and size
%==========================================================================

shocks;
var me_w;    stderr	sqrt(0.114)		;
var me_E;  	 stderr	sqrt(0.051)		;
var me_pi_d; stderr	sqrt(0.15)		;
var me_pi_i; stderr	sqrt(0.184)		;
var me_y;  	 stderr	sqrt(0.036)		;
var me_c; 	 stderr	sqrt(0.053)		;
var me_i;  	 stderr	sqrt(0.346)		;
var me_imp;  stderr	sqrt(1.364)		;
var me_ex;   stderr	sqrt(1.925)		;
var me_ystar;stderr	sqrt(0.041)		;
var eps_g;   stderr 0.98;

%---------------------------------------------------------------------------------------------------
end;
%---------------------------------------------------------------------------------------------------

%==========================================================================
% Define observable variables
%==========================================================================
varobs 

R_                      %domestic policy interest rate
pi_c_					%CPI
pi_cbar_                %inflation target
pi_i_                   %investment inflation
pi_d_                   %domestic goods inflation
dy_                     %change in real GDP
dc_                     %change in consumption
di_                     %change in investment
dimp_                   %change in import
dex_                    %change in export
dy_star_                %change in foreign real GDP
pi_star_                %change in foreign inflation rate
R_star_                 %foreign policy interest rate
dE_                     %change in employment
dS_                     %change in nominal exchange rate (rather than real exchange rate)
dw_                     %change in employee compensation

;

%==========================================================================
% Define parameters to be estimated using Beyesian method
%==========================================================================
estimated_params;

%---------------------------------------------------------------------------------------------------
% Adjustment costs
%---------------------------------------------------------------------------------------------------

phi_i,          normal_pdf, 8, 1.5;

%---------------------------------------------------------------------------------------------------
% Consumption
%---------------------------------------------------------------------------------------------------
b,              beta_pdf, 0.7, 0.1;

%---------------------------------------------------------------------------------------------------
% Calvo parameters
%---------------------------------------------------------------------------------------------------
theta_d,        beta_pdf, 0.9, 0.05;
theta_mc,       beta_pdf, 0.9, 0.05;
theta_mi,       beta_pdf, 0.76, 0.05;
theta_x,        beta_pdf, 0.5, 0.05;
theta_e,        beta_pdf, 0.9, 0.05;

%---------------------------------------------------------------------------------------------------
% Indexation
%---------------------------------------------------------------------------------------------------
xi_d,        beta_pdf, 0.6, 0.1;
xi_mc,       beta_pdf, 0.6, 0.1;
xi_mi,       beta_pdf, 0.6, 0.1;

%---------------------------------------------------------------------------------------------------
% Exchange rate
%---------------------------------------------------------------------------------------------------
phi_a,          inv_gamma_pdf, 0.04, 1;
phi_s,          inv_gamma_pdf, 0.6, 1; 

%---------------------------------------------------------------------------------------------------
% Taylor rules
%---------------------------------------------------------------------------------------------------
rho_r,            beta_pdf, 0.8, 0.05; 
phi_pi,           gamma_pdf, 1, 0.15; 
phi_dpi,          gamma_pdf, 0.1, 0.05; 
phi_y,            gamma_pdf, 0.5, 0.1;  
phi_dy,           gamma_pdf, 0.25, 0.1;  
phi_x,            normal_pdf, 0.01, 0.05;  

%---------------------------------------------------------------------------------------------------
% Persistence parameters
%---------------------------------------------------------------------------------------------------
rho_mu_z,       beta_pdf, 0.7, 0.1;
rho_z,          beta_pdf, 0.7, 0.1;
rho_i,          beta_pdf, 0.7, 0.1;
rho_ztildestar, beta_pdf, 0.7, 0.1;
rho_c,          beta_pdf, 0.7, 0.1;
rho_H,          beta_pdf, 0.7, 0.1;
rho_a,          beta_pdf, 0.7, 0.1;
rho_lambda_d,   beta_pdf, 0.7, 0.1;
rho_lambda_mc,  beta_pdf, 0.7, 0.1;
rho_lambda_mi,  beta_pdf, 0.7, 0.1;
rho_lambda_x,   beta_pdf, 0.7, 0.1;

%---------------------------------------------------------------------------------------------------
% Foreign Economy parameters
%---------------------------------------------------------------------------------------------------
rho_Rstar,       beta_pdf, 0.7, 0.1;
rho_pistar,      beta_pdf, 0.7, 0.1;
rho_ystar,       beta_pdf, 0.7, 0.1;


%---------------------------------------------------------------------------------------------------
% Structural shocks
%---------------------------------------------------------------------------------------------------
stderr eps_mu_z,inv_gamma_pdf, 0.3, inf;
stderr eps_z,   inv_gamma_pdf, 0.3, inf;
stderr eps_i,   inv_gamma_pdf, 0.3, inf;
stderr eps_z_tildestar, inv_gamma_pdf, 0.3, inf;
stderr eps_c,   inv_gamma_pdf, 0.3, inf;
stderr eps_H,   inv_gamma_pdf, 0.3, inf;
stderr eps_a,   inv_gamma_pdf, 0.3, inf; 
stderr eps_d,   inv_gamma_pdf, 0.3, inf;
stderr eps_mc,  inv_gamma_pdf, 0.3, inf;
stderr eps_mi,  inv_gamma_pdf, 0.3, inf;
stderr eps_x,   inv_gamma_pdf, 0.3, inf;
stderr eps_r,   inv_gamma_pdf, 0.3, inf; 
stderr eps_pi_cbar,     inv_gamma_pdf, 0.3, inf;
stderr eps_ystar,       inv_gamma_pdf, 0.3, inf; 
stderr eps_pistar,      inv_gamma_pdf, 0.3, inf; 
stderr eps_Rstar,       inv_gamma_pdf, 0.3, inf; 
stderr eps_i_k,inv_gamma_pdf, 0.001, inf;
stderr eps_i_y,inv_gamma_pdf, 0.001, inf;

%---------------------------------------------------------------------------------------------------
end;
%---------------------------------------------------------------------------------------------------

%==========================================================================
estimation(order=1,datafile=us_data,  first_obs=1, nobs=105, plot_priors=0, mh_replic=40000, mh_jscale=0.33, mh_nblocks=3, optim = ('NumberOfMh', 20), mode_compute=6, mode_check); %mode_compute=1, dpss_data_12q4_full, us_data
%==========================================================================

%==========================================================================
stoch_simul(irf=20, periods=200, simul_replic = 1000) R_ pi_c_ pi_cbar_ dy_ dc_ di_ dy_star_ dE_ pi_star_ R_star_ dS_ dex_ dimp_ dw_ pi_i_ pi_d_ ;
%==========================================================================
