data_raw =xlsread('macro-data-Non-Transformées.xlsx');
%%
clc
data = data_raw(301:end-3,:); %on prend les données à partir de 1996
data = data(:,~isnan(data(1,:))); %supprime les colonnes où le 1er élément est NaN
N = size(data,2) ;
data = [data(:,1:12) data(:,14:N)]; %on enlève la colonne 13 (contient des nbrs négatifs)(à faire sur excel directement)

T = size(data,1); N = size(data,2) ; 
tcodes =data(T,:); 
dataTransf = zeros(T,N) ; 

%    -- Tcodes:
%             0 First Difference
%             1 Log-Level
%             2 Log-First-Difference
%             3 Log-Second-Difference
%% passage en log diff de toutes les séries 
 for i = 1:N
%     
%         if tcodes(1,i) ==0 % 1st differnce
%            dataTransf(2:T,i)=diff(data(:,i));
%          
%         elseif tcodes(1,i) == 1 %log
%             dataTransf(:,i) = log(data(:,i));
%             
%         elseif tcodes(1,i) == 2 %log difference
            dataTransf(2:T,i) = diff(log(data(:,i)));
%             
%         elseif tcodes(1,i) == 3 %log difference
%             dataTransf(3:T,i) = diff(log(data(:,i)),2);
%         end
%         
 end

%% tests de stationnarité 
O = length(dataTransf);
for i = 1:N
    [h(i),pval1(i)] = adftest(dataTransf(:,i), 'model','ts', 'lags',2);
end 
%1 : série stationnaire

%%
[COEFF, SCORE, LATENT] = pca(dataTransf); 

%ici j'ai repris ce qu'on avait fait avec Lepen dans le TD3
cov_data = cov(dataTransf); 
[V,D] = eig(cov_data);
val_propres = diag(D);
[val_propres, I] =sort(diag(D),'descend'); % tri des valeurs propres dans l'ordre décroissant
V = V(:,I');
contribution = val_propres/sum(val_propres);% calcul des contributions de chaque composante principale ? la somme de la variance des rendements.
cum_cont = cumsum(contribution); % calcul des contribution cumulée valeurs propres
plot(cum_cont)