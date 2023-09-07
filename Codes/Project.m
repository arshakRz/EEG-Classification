%% CI Project 98106531 Arshak Rezvani
%% 
clc; close all; clear all; 

% Loading the data
load('CI_Project_data.mat');

% Plotting the first channel of the first observation
Fs = 256;
t = 1/Fs:1/Fs:1.5;
plot(t, TrainData(1,:,1));
xlabel('t(s)');
title('The First Channel of The First Observation');
grid minor 
%%
% Extraction of features from the data (Train and Test)
stat_features_tot = zeros(160, 1245);
freq_features_tot = zeros(160, 300);
data_tot = cat(3, TrainData,TestData);
edges = [-20 -12 -8 -4 -2 -1 -0.5 0 0.5 1 2 4 8 12 20];
for i = 1:size(data_tot,3)
    data = data_tot(:,:,i);

    % Statistical Features

    % Variance
    var_data = var(data, 0, 2);
    stat_features_tot(i,1:30) = var_data;

    % Correlation 
    cor_data = corrcoef(data');
    m = 1;
    for j = 1:30
        for k = j+1:30
            stat_features_tot(i,30+m) = cor_data(k,j);
            m = m+1;
        end
    end
    
    % Kurtosis
    kurtosis_data = kurtosis(data,1,2);
    stat_features_tot(i,466:495) = kurtosis_data;
    
    % Skewness
    skewness_data = skewness(data,1,2);
    stat_features_tot(i,496:525) = skewness_data;

    % AR Model Coefficients of 30 Channels
    for j = 1:30
        time_series = data(j,:);
        ar_time_series = ar(time_series,10);
        stat_features_tot(i,(526+(j-1)*10):525+(j)*10) = ar_time_series.A(2:11);       
    end

    % Histogram of 30 Channels
    for j = 1:30
        time_series = data(j,:);
        N = histcounts(time_series,edges);
        stat_features_tot(i,(826+(j-1)*14):825+(j)*14) = N;
    end 
    
    % Frequency Domain Features

    % Calculating the Frequency and Time Vectors
    df = (Fs)/(length(t)-1);
    f = -Fs/2:df:Fs/2;

    % Max Freq
    max_freq = zeros(1,30);
    for j = 1:30
        [idc, max_freq(j)] = max(abs(fftshift(fft(data(j,:))*1/Fs)));
    end
    freq_features_tot(i,1:30) = abs(f(max_freq));

    % Mean and Median Freq
    mean_freq = zeros(1,30);
    med_freq = zeros(1,30);
    for j = 1:30
        mean_freq(j) = meanfreq(data(j,:),Fs);
        med_freq(j) = medfreq(data(j,:),Fs);
    end
    freq_features_tot(i,31:60) = mean_freq;
    freq_features_tot(i,61:90) = med_freq;
    
    % Power Spectral Ratio
    for j = 1:30
        tot = 193:384;
        time_series = data(j,:);
        fft_time_series = fftshift(fft(data(j,:))*1/Fs);
        fft_tot = fft_time_series(tot);
        tot_psd = abs(fft_tot).^2;
        tot_power = sum(tot_psd)*df;

        % Calculating power of each band 
        delta = 193:197;
        fft_delta = fft_time_series(delta);
        delta_psd = abs(fft_delta).^2;
        delta_power = sum(delta_psd)*df;
        theta = 198:203;
        fft_theta = fft_time_series(theta);
        theta_psd = abs(fft_theta).^2;
        theta_power = sum(theta_psd)*df;
        alpha = 204:211;
        fft_alpha = fft_time_series(alpha);
        alpha_psd = abs(fft_alpha).^2;
        alpha_power = sum(alpha_psd)*df;
        l_beta = 211:215;
        fft_l_beta = fft_time_series(l_beta);
        l_beta_psd = abs(fft_l_beta).^2;
        l_beta_power = sum(l_beta_psd)*df;
        m_beta = 216:223;
        fft_m_beta = fft_time_series(m_beta);
        m_beta_psd = abs(fft_m_beta).^2;
        m_beta_power = sum(m_beta_psd)*df;
        h_beta = 224:238;
        fft_h_beta = fft_time_series(h_beta);
        h_beta_psd = abs(fft_h_beta).^2;
        h_beta_power = sum(h_beta_psd)*df;
        gamma = 238:384;
        fft_gamma = fft_time_series(gamma);
        gamma_psd = abs(fft_gamma).^2;
        gamma_power = sum(gamma_psd)*df;
        freq_features_tot(i,(j-1)*7+91) = delta_power/tot_power;
        freq_features_tot(i,(j-1)*7+92) = theta_power/tot_power;
        freq_features_tot(i,(j-1)*7+93) = alpha_power/tot_power;
        freq_features_tot(i,(j-1)*7+94) = l_beta_power/tot_power;
        freq_features_tot(i,(j-1)*7+95) = m_beta_power/tot_power;
        freq_features_tot(i,(j-1)*7+96) = h_beta_power/tot_power;
        freq_features_tot(i,(j-1)*7+97) = gamma_power/tot_power;
    end
end

% Concatinating the stat. and freq. features
features_tot = [stat_features_tot freq_features_tot];

%% Saved The Features Calculated From the Last Section 
% Comment the 3 lines below to recalculate the features

clc; close all; clear all;
load('CI_Project_data.mat');
load('features_tot.mat');

% Normalizing the Features
[features_norm, PS] = mapstd(features_tot(1:120,:)',0,1); 
features_norm = features_norm';
features_norm_test = mapstd('apply',features_tot(121:160,:)',PS);
features_norm_test = features_norm_test';

% Feature ranking using fisher score(1d)
%score_fisher = fisher_score(features_norm, TrainLabel');
%[~, ranking_fisher] = maxk(score_fisher,1545);

% Feature ranking using chi-square tests
[ranking_chi, score_chi] = fscchi2(features_norm,TrainLabel');

% Feature ranking  using minimum redundancy maximum relevance (MRMR) algorithm
[ranking_mrmr, score_mrmr] = fscmrmr(features_norm,TrainLabel');
%%
% Calculating the first 30 principal components of the features matrix
loadings = pca(features_norm);
top30 = features_norm*loadings(:,1:30);
loadings_test = pca(features_norm_test);
top30_test = features_norm_test*loadings_test(:,1:30);

% Choosing top 10 set of features from pca using genetic algorithm 
options = optimoptions('ga');
options = optimoptions(options,'CrossoverFcn', @crossoverintermediate);
options = optimoptions(options,'FitnessScalingFcn', @fitscalingprop);
options = optimoptions(options,'SelectionFcn', @selectiontournament);
options = optimoptions(options,'MutationFcn', @mutationpower);
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'PlotFcn', {  @gaplotbestf @gaplotscorediversity @gaplotscores });
[result, fval] = ga(@fisher_score_nd,20,[],[],ones(1,20),10,zeros(1,20),ones(1,20),[],1:20,options);
ranking_ga = find(result==1);

% Concatenating the Ranked Features and Choosing the Top 10 
rank_feat = [ranking_fisher; ranking_chi; ranking_mrmr];
rank_feat = rank_feat(:,1:10);
rank_feat = [rank_feat; ranking_ga];

% Creating the Training Target
Y = zeros(2,120) ;
Y(1,TrainLabel==1) = 1 ;
Y(2,TrainLabel==2) = 1 ;

% 5-fold cross-validation for choosing the top set of training features and
% network specifications 
indices = crossvalind('Kfold',TrainLabel,5);

%%
% Finding the best network and features sets using one layer MLP
% The number of input features because low number of training data in set to 10
ACC_MLP = zeros(4, 20);
accu = 0;
accu_ga = 0;
for j = 1:4
    feat = [rank_feat(j,:)];
    X = [features_norm(:,feat)];
    if j == 4
        X = [top30(:,feat)];
    end
    X = X';
    for n = 1:20
        ACC = 0;
        for i = 1:5
            net = patternnet(n);
            net.divideParam.trainRatio = 1;
            net.divideParam.valRatio = 0;
            net.divideParam.testRatio = 0;
            TrainX = X(:,indices~=i);
            ValX = X(:,(indices==i));
            TrainY = Y(:,(indices~=i));
            ValY = Y(:,(indices==i));
            net = train(net,TrainX,TrainY);
            predict_y = net(ValX);
            [~,mindx] = max(predict_y);
            p_ValY = zeros(2,24);
            p_ValY(1,(mindx==1)) = 1;
            p_ValY(2,(mindx==2)) = 1;
            ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:)));
        end
        ACC_MLP(j,n) = ACC/120;
        if (ACC/120>accu && j~=4)
            feat = [rank_feat(j,:)];
            X = [features_norm(:,feat)];
            X = X';
            net = patternnet(n);
            net.divideParam.trainRatio = 1;
            net.divideParam.valRatio = 0;
            net.divideParam.testRatio = 0;
            net = train(net,X,Y);   
            feat_test = [rank_feat(j,:)];
            X_test = [features_norm_test(:,feat_test)];
            X_test = X_test';
            predict_y_test = net(X_test);
            [~,TestLabel_MLP] = max(predict_y_test);
            accu = ACC/120;
        end
        if (ACC/120>accu_ga && j==4)
            feat = [rank_feat(j,:)];
            X = [top30(:,feat)];
            X = X';
            net = patternnet(n);
            net.divideParam.trainRatio = 1;
            net.divideParam.valRatio = 0;
            net.divideParam.testRatio = 0;
            net = train(net,X,Y);
            X_test = [top30_test(:,rank_feat(4,:))];
            X_test = X_test';
            predict_y_test = net(X_test);
            [~,TestLabel_MLP_ga] = max(predict_y_test);
            accu_ga = ACC/120;
        end
    end
end
%%
% Finding the best network and features sets using RBF
% The number of input features because low number of training data in set to 10
spreadMat = [1.5,3,4.5,6,7.5,9];
NMat = [5,10,15,20,25] ;
ACC_RBF = zeros(4, 6, 5);
accu = 0;
accu_ga = 0;
for j = 1:4
    feat = [rank_feat(j,:)];
    X = [features_norm(:,feat)];
    if j == 4
        X = [top30(:,feat)];
    end
    X = X';
    for s = 1:6
        spread = spreadMat(s) ;
        for n = 1:5
            Maxnumber = NMat(n) ;
            ACC = 0;
            for i = 1:5
                TrainX = X(:,indices~=i);
                ValX = X(:,(indices==i));
                TrainY = Y(:,(indices~=i));
                ValY = Y(:,(indices==i));
                net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
                predict_y = net(ValX);
                [maxval,mindx] = max(predict_y) ;
                p_ValY = zeros(2,24) ;
                p_ValY(1,(mindx==1)) = 1 ;
                p_ValY(2,(mindx==2)) = 1 ;
                ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:)));
            end
            if (ACC/120>accu && j~=4)
                feat = [rank_feat(j,:)];
                X = [features_norm(:,feat)];
                X = X';
                net = newrb(X,Y,10^-5,spread,Maxnumber) ;
                feat_test = [rank_feat(j,:)];
                X_test = [features_norm_test(:,feat_test)];
                X_test = X_test';
                predict_y_test = net(X_test);
                [~,TestLabel_RBF] = max(predict_y_test);
                accu = ACC/120;
            end
            ACC_RBF(j,s,n) = ACC/120;
        end
    end
end
spreadMat_ga = [4.5,6,7.5,9,10.5,12,13.5,15,16.5,18];
ACC_RBF_ga = zeros(10, 5);
feat = [rank_feat(4,:)];
X = [top30(:,feat)];
X = X';
for s = 1:10
    spread = spreadMat_ga(s) ;
    for n = 1:5
        Maxnumber = NMat(n) ;
        ACC = 0;
        for i = 1:5
            TrainX = X(:,indices~=i);
            ValX = X(:,(indices==i));
            TrainY = Y(:,(indices~=i));
            ValY = Y(:,(indices==i));
            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
            predict_y = net(ValX);
            [maxval,mindx] = max(predict_y) ;
            p_ValY = zeros(2,24) ;
            p_ValY(1,(mindx==1)) = 1 ;
            p_ValY(2,(mindx==2)) = 1 ;
            ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:)));
        end
        if (ACC/120>accu_ga)
            feat = [rank_feat(4,:)];
            X = [top30(:,feat)];
            X = X';
            net = newrb(X,Y,10^-5,spread,Maxnumber) ;
            feat_test = [rank_feat(4,:)];
            X_test = [top30_test(:,feat_test)];
            X_test = X_test';
            predict_y_test = net(X_test);
            [~,TestLabel_RBF_ga] = max(predict_y_test);
            accu_ga = ACC/120;
        end
        ACC_RBF_ga(s,n) = ACC/120;
    end
end

%% Showing and Plotting The Output
close all
labels = [TestLabel_MLP; TestLabel_MLP_ga; TestLabel_RBF; TestLabel_RBF_ga];
figure
sgtitle('5-Fold Cross Validation MLP Accuracy With The Top 10 Features Calculated With:') 
subplot(2,2,1);
bar(1:20,ACC_MLP(1,:)*100);
ylim([0 100]);
title('1D Fisher Score');
xlabel('Number of Hidden Layer Neurons')
ylabel('Accuracy(%)')

subplot(2,2,2);
bar(1:20,ACC_MLP(2,:)*100);
ylim([0 100]);
title('Chi-Squared Test');
xlabel('Number of Hidden Layer Neurons')
ylabel('Accuracy(%)')

subplot(2,2,3);
bar(1:20,ACC_MLP(3,:)*100);
ylim([0 100]);
title('MRMR Algorithm');
xlabel('Number of Hidden Layer Neurons')
ylabel('Accuracy(%)')

subplot(2,2,4);
bar(1:20,ACC_MLP(4,:)*100);
ylim([0 100]);
title('PCA and Genetic Algorithm');
xlabel('Number of Hidden Layer Neurons')
ylabel('Accuracy(%)')

figure
sgtitle('5-Fold Cross Validation RBF Accuracy With The Top 10 Features Calculated With:') 

subplot(2,2,1);
bar3(squeeze(ACC_RBF(1,:,:))*100);
zlim([0 100]);
title('1D Fisher Score');
set(gca,'YTickLabel', spreadMat);
set(gca,'XTickLabel', NMat);
ylabel('\sigma','Interpreter','tex','FontSize',14);
xl = xlabel('Number of Hidden Layer Neurons','Rotation',20,'FontSize',10);
xl.Position=xl.Position+[1.7 -0.7 0];    
zlabel('Accuracy(%)')


subplot(2,2,2);
bar3(squeeze(ACC_RBF(2,:,:))*100);
zlim([0 100]);
title('Chi-Squared Test');
set(gca,'YTickLabel', spreadMat);
set(gca,'XTickLabel', NMat);
ylabel('\sigma','Interpreter','tex','FontSize',14);
xl = xlabel('Number of Hidden Layer Neurons','Rotation',20,'FontSize',10);
xl.Position=xl.Position+[1.7 -0.7 0];    
zlabel('Accuracy(%)')

subplot(2,2,3);
bar3(squeeze(ACC_RBF(3,:,:))*100);
zlim([0 100]);
title('MRMR Algorithm');
set(gca,'YTickLabel', spreadMat);
set(gca,'XTickLabel', NMat);
ylabel('\sigma','Interpreter','tex','FontSize',14);
xl = xlabel('Number of Hidden Layer Neurons','Rotation',20,'FontSize',10);
xl.Position=xl.Position+[1.7 -0.7 0];    
zlabel('Accuracy(%)')

subplot(2,2,4);
bar3(squeeze(ACC_RBF(4,:,:))*100);
zlim([0 100]);
title('PCA and Genetic Algorithm');
set(gca,'YTickLabel', spreadMat);
set(gca,'XTickLabel', NMat);
ylabel('\sigma','Interpreter','tex','FontSize',14);
xl = xlabel('Number of Hidden Layer Neurons','Rotation',20,'FontSize',10);
xl.Position=xl.Position+[1.7 -0.7 0];    
zlabel('Accuracy(%)')

figure
bar3(ACC_RBF_ga(:,:)*100);
zlim([0 100]);
title('5-Fold Cross Validation RBF Accuracy With Higher Distance Function Radius and Top 10 Features Calculated With PCA and Genetic Algorithm');
set(gca,'YTickLabel', spreadMat_ga);
set(gca,'XTickLabel', NMat);
ylabel('\sigma','Interpreter','tex','FontSize',14);
xl = xlabel('Number of Hidden Layer Neurons','Rotation',20,'FontSize',10);
xl.Position=xl.Position+[1.7 -0.7 0];    
zlabel('Accuracy(%)')