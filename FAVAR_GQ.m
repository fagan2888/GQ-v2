%FAVAR CODE

clear all;
clc;

%% USER INPUT

% Select a subset of the 6 variables in Y

subset = 2;     % 1: DF1, DF2 (2 vars)
                % 2: FFR, DF1, DF2 (3 vars)
                % 3: Indus. Prod., Inflation CPI, S&P 500, 10Y TB, DF1, DF2 (6 vars)
%% test sur les data de SW 
%test = xlsread('data v2.xlsx','SW only data');
%tcode = xlsread('data v2.xlsx','SW tcode');
%t = size(test,2);

%for i = 1:t
%    y(:,i)=transx(test(:,i),tcode(i)); 
%end 

%y = xlsread('data v2.xlsx','SW trans');

%for i = 1:t
%    [h(i),p(i)] = adftest(y(:,i));
%end

%[~,~,eigenvalues,~,percentage] = pca(y);
%screeplot(y,60);

%% DATA

% ----- LOAD DATA -----
% X : factors 
xdata = xlsread('data v2.xlsx','X Lucille');

% Transformation de X
for i = 1:size(xdata,2)
    xtemp(:,i) = diff(log(xdata(:,i)));
end

xdata = xtemp;

% Y 
ydata = xlsread('data v2.xlsx','Y 2007');

if subset == 1
    ydata = ydata(:,[1 2]);
elseif subset == 2
    ydata = ydata(:,[1 2 3]);
elseif subset == 3
    ydata = ydata(:,[1 2 3 4 5 6]);  
end

% Slow/Fast
slowcode = xlsread('data v2.xlsx','slow');
% Dates
yearlab = xlsread('data v2.xlsx','year 2007');
% Names de X
[~,namesX] = xlsread('data v2.xlsx','X Lucille');
namesX = namesX(1,2:end)';


% Size
t0 = size(xdata,1); % nb of periods
t1 = size(xdata,2); % nb of factors
t2 = size(ydata,2); % nb of explained variables

% ----- Description de Y (Graphiques) -----

y_non_trans = xlsread('data v2.xlsx','Y 2007 non transformés');

% FED Rates
figure(1)
plot(yearlab,y_non_trans(:,1),'Color','black','LineWidth',2);
grid on;
xlabel('Time');
ylabel('Federal Funds Rates');

% US Loans
figure(2)
plot(yearlab,y_non_trans(:,2),'Color','black','LineWidth',2);
grid on;
xlabel('Time');
ylabel('Loans US default rates');

% US bonds
figure(3)
plot(yearlab,y_non_trans(:,3),'Color','black','LineWidth',2);
grid on;
xlabel('Time');
ylabel('US corp bonds Default rates');

% ----- First test de stationnarité ADF ----- 
% Y
for i = 1:size(ydata,2)
    [hY1(i),pval1(i)] = adftest(ydata(:,i));
end 

% X 
for i = 1:size(xdata,2)
    [hY2(i),pval2(i)] = adftest(xdata(:,i));
end 

% Transformation pour stationnariser les séries 
% On effectue pas de transformation ici vu que les variables le sont déjà
% quand on les importe. 

% trans = xlsread('data v2.xlsx','trans');

%pour X
%for i_x = 1:size(xdata,2)   % Transform "X"
%    xtempraw(:,i_x) = transx(xdata(:,i_x),tcode(i_x)); %#ok<AGROW>
%end

%pour Y
%for i_y = 1:size(ydata,2)
%    ytempraw(:,i_y) = transx(ydata(:,i_y),2);
%end

%xdata = xtempraw;
%ydata = ytempraw;

% Test de stationnarité ADF après transformations
% Y
%for i = 1:size(ydata,2)
%    [hY2(i),pval2(i)] = adftest(ydata(:,i));
%end 

% X
%for i = 1:size(ydata,2)
%    [hX2(i),pvalX2(i)] = adftest(xdata(:,i));
%end 


% Define X et Y
X = xdata;   % Factors
Y = ydata; % Y 
namesXY = [namesX ; 'FFR'; 'US Loans DF' ; 'US Bonds DF']; % Noms


%------ Low frequency trends ------%

% Pour lisser les data
% Removing low frequency trends using a Bi-Weight trend
bw_bw = 100; % Bi-Weight Parameter for local demeaning
for is = 1:t2; 
    tmp = bw_trend(Y(:,is),bw_bw);
   	Y_trend(:,is)= tmp;
   	Y(:,is) = Y(:,is) - Y_trend(:,is); 
end;

for is = 1:t1; 
    tmp = bw_trend(X(:,is),bw_bw);
   	X_trend(:,is)= tmp;
   	X(:,is) = X(:,is) - X_trend(:,is); 
end;

%------ Standardization ------%
ymean = nanmean(Y)';                                        % mean (ignoring NaN)
mult = sqrt((sum(~isnan(Y))-1)./sum(~isnan(Y)));     % num of non-NaN entries for each series
ystd = (nanstd(Y).*mult)';                                  % std (ignoring NaN)
Y = (Y - repmat(ymean',t0,1))./repmat(ystd',t0,1);  % standardized data

xmean = nanmean(X)';                                        
mult = sqrt((sum(~isnan(X))-1)./sum(~isnan(X)));     
xstd = (nanstd(X).*mult)';                                  
X = (X - repmat(xmean',t0,1))./repmat(xstd',t0,1); 

% Faut-il standardiser X ?

% Number of observations and dimension of X and Y
T=size(Y,1); % T time series observations
N=size(X,2); % N series from which we extract factors
M=size(Y,2); % Taille de Y


%% FAVAR


%------ Lag Selection ------%

[InformationCriterion, aicL, bicL, hqcL, fpeL, CVabs, CVrel, CVrelTr] = AicBicHqcFpe(Y, 12);
p =CVrel;
% Ici ils appliquent la méthode de cross validation (voir les ref)

% Comment faisons-nous pour déterminer le nb de retards ? 
% PCA ?

[~,~,C,~,pca_perc] = pca(X);

% ----- Number of factors ----- 

% Cross Validation : voir papier de recherche qui est en référence dans le
% mémoire des étudiants
constant = 1;
[numFt, FAVAR_CV_RMSE, CVabs, CVrel] = favarTuneFactorNum(X, Y(:, :), p, 12, constant);

% Onatski(2010)
numFtOnatski = onatski(X,12);

% Bai & Ng
% à voir 
BG = NbFactors(X);

% Autre méthode : ici ils font avec une interprétation économique. PCA ?
% Technique de Kaiser
% Technique du coude 
% Screeplot 
% Voir comment on peut interpréter le pca 

screeplot(X, 20);
% Nb factors = 6

%------ Factors estimation ------

F = factorsFAVAR(X,6);

% Même chose mais en mieux 
[~,score] = pca(X);
F = score(:,1:6);


%------ FAVAR estimation ------

% Data
DATA_FAVAR = [Y F];

% Parameters
%p = CVrel;                  % number of lags
constant = 1;              % include the constant
hor = 20;                  % horizon
iter = 500;                % number of iterations
conf = [90 60];            % level for confidence bands

% Estimation of the FAVAR
Cf_FAVAR = ir(DATA_FAVAR,p,constant,hor,'cholimpact');
[ULbf_FAVAR,~,~,~,ULb2f_FAVAR] = bootbands(DATA_FAVAR,p,constant,iter,hor,conf,'cholimpact');

% Plot resutls
% A MODIFIER
irfs_FAVAR = permute(Cf_FAVAR,[3 1 2]);
irfs_FAVAR = irfs_FAVAR(:,:,1);
conf_FAVAR1 = permute(ULbf_FAVAR,[3 1 2 4]);
conf_FAVAR_down1 = conf_FAVAR1(:,:,1,1);
conf_FAVAR_up1 = conf_FAVAR1(:,:,1,2);
conf_FAVAR2 = permute(ULb2f_FAVAR,[3 1 2 4]);
conf_FAVAR_down2 = conf_FAVAR2(:,:,1,1);
conf_FAVAR_up2 = conf_FAVAR2(:,:,1,2);
plot_titles = { 'Oil Price', 'Real GDP' , 'Total Employment', 'Inflation'};

figure(7);
for ii = 1:3
subplot(2,2,ii); 
% IRFs of the SVAR
%plot(irfs_SVAR(:,ii),'red','LineWidth',1.5,'LineStyle','-');hold on; 
% IRFs of the FAVAR
plot(irfs_FAVAR(:,ii),'black','LineWidth',2);title(plot_titles(:,ii));  
%formatGraph('Quarter', '');
% Horizontal line at 0
hold on;plot(xlim, [0 0], 'color', 0.5*[1 1 1],'LineWidth',1.1);                     
% 90% confidence interval
hold on;plot(conf_FAVAR_down1(:,ii),'black','LineStyle',':');hold on;plot(conf_FAVAR_up1(:,ii),'black','LineStyle',':')
% 60% confidence interval
hold on;plot(conf_FAVAR_down2(:,ii),'black','LineStyle','-');hold on;plot(conf_FAVAR_up2(:,ii),'black','LineStyle','-')
end

% Sortir la contribution des variables à chaque composante principale 
% IRF avec mgoretti
% Slow/fast crtieria et comment les utiliser 
% Les tests avec les contraintes 
% Table 11 : Dynamic/static factors
% Table 12 : 48 Months Forecast Error Decompositions
% Results of Exclusions Restrictions tests 
