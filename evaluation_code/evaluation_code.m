%   Distribution code Version 1.0 -- Oct 12, 2013 by Cewu Lu 
%
%   The Code can be used to evaluate your detection results in our Avenue Dataset, build based on  
%   [1] "Abnormal Event Detection at 150 FPS in Matlab" , Cewu Lu, Jianping Shi, Jiaya Jia, 
%   International Conference on Computer Vision, (ICCV), 2013
%   
%   The code and the algorithm are for non-comercial use only.

testFileNum = 21;
overlapThr = 0.5;

%% generating a random guess result
% produced result are save in document "rand_guess"
% detected mask of x th testing video using random guess is save x_rand.mat  
% volRand{ii} is the detected mask of ii frame 

for idx = 1 : testFileNum
    load(['testing_label_mask\', num2str(idx), '_label.mat'], 'volLabel'); 
    volRand = {};
    
    [Hs, Ws] = size(volLabel{1});
    for ii = 1 : length(volLabel) 
        volRand{ii} = boolean(zeros(Hs, Ws));
        sx = randi(Hs);
        ex = randi(Hs);
        sy = randi(Ws);
        ey = randi(Ws);        
        sx  = min(sx,ex); ex = max(sx,ex);
        sy  = min(sy,ey); ey = max(sy,ey);
        volRand{ii}(sx:ex,sy:ey) =  boolean(1);
    end
    fprintf('random guess in %d th video \n', idx);
    save(['rand_guess\', num2str(idx), '_rand.mat'], 'volRand'); 
end



%% test your result
% ground truths are saved in document "testing_label_mask"
% ground truth mask of x th testing video using random guess is saved x_label.mat  
% volLabel{ii} is the detected mask of ii th frame 

acc = zeros(1, testFileNum);
for idx = 1 : testFileNum
    load(['testing_label_mask\', num2str(idx), '_label.mat'], 'volLabel');
    load(['rand_guess\',num2str(idx),'_rand.mat'], 'volRand');
    ratios = zeros(1,length(volLabel));
    for ii = 1 : length(volLabel) 
        unionSet = sum(sum(volRand{ii}|volLabel{ii}));
        interSet = sum(sum(volRand{ii}&volLabel{ii}));
        if unionSet == 0
            ratios(ii) = 1;
        else
            ratios(ii) = interSet/unionSet;
        end
    end
    acc(idx) = sum(ratios > overlapThr)/length(ratios);
    fprintf('Accuracy in %d th video is %.1f %% \n', idx, 100*acc(idx));
end
fprintf('random guess overall accuracy is %.2f %% \n', 100*mean(acc)); 

