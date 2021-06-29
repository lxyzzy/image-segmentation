clear all;close all;clc;
img = double(rgb2gray(imread('lena.jpg')));
%设置聚类数
cluster_num = 5;
% 设置初始均值和方差，均值为随机在各色块中选点并取整所得
mu=[50 70 100 190 220];
sigma = 15^2*ones(1,cluster_num);
pw = zeros(cluster_num,size(img,1)*size(img,2));
pc = rand(1,cluster_num);
pc = pc/sum(pc);%将类概率归一化
max_iter = 50;%以迭代次数来作为停止的条件
iter = 1;
while iter <= max_iter
    %----------E-------------------
    for i = 1:cluster_num
        MU = repmat(mu(i),size(img,1)*size(img,2),1);
        %高斯模型
        temp = 1/sqrt(2*pi*sigma(i))*exp(-(img(:)-MU).^2/2/sigma(i));
        temp(temp<0.000001) = 0.000001;%防止出现0
        pw(i,:) = pc(i) * temp;
    end
    pw = pw./(repmat(sum(pw),cluster_num,1));%归一
    %----------M---------------------
    %更新参数集
    for i = 1:cluster_num
         pc(i) = mean(pw(i,:));
         mu(i) = pw(i,:)*img(:)/sum(pw(i,:));
         sigma(i) = pw(i,:)*((img(:)-mu(i)).^2)/sum(pw(i,:));
    end
    %------------show-result---------------
    [~,label] = max(pw);
    %改大小
    label = reshape(label,size(img));
    imshow(label,[])
    title(['iter = ',num2str(iter)]);
    pause(0.1);
    M(iter,:) = mu;
    S(iter,:) = sigma;
    
    % 录制gif
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if iter == 1
        imwrite(I,map,'test_gray.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test_gray.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    iter = iter + 1;
end
%将均值与方差的迭代过程显示出来
figure
for i = 1:cluster_num
    plot(M(:,i));
    hold on
end
title('均值变化过程');
figure
for i = 1:cluster_num
    plot(S(:,i));
    hold on
end
title('方差变化过程');
第二部分
clear all;close all;clc;
img = double(imread('lena.jpg'));
N=size(img,1)*size(img,2);
%设置聚类数
cluster_num = 5;
% mu为各聚类的RGB三通道均值，每列为一各聚类的均值，
% 初始值是在图中黄、绿、红、白、紫部分随机取一点得到的RGB值
mu=[255,115,216,247,75;...
        220,124,50,215,45;...
        50,31,39,185,75];
% sigma为各聚类三通道的方差，初始方差均设为20^2
sigma=20^2*ones(3,cluster_num);
pw = zeros(N,cluster_num); %各像素点属于各聚类的概率，最后需要对聚类归一化
pc = rand(1,cluster_num);% 各聚类总体的概率分布
pc = pc/sum(pc);%将类概率归一化
max_iter = 30;%以迭代次数来作为停止的条件
iter = 1;
while iter <= max_iter
    %----------E-------------------
    for i = 1:cluster_num
        %矩阵操作--速度快
        MU = repmat(reshape(mu(:,i),1,3),N,1);
        %高斯模型
        sigmaM=repmat(reshape(sigma(:,i),1,3),N,1);
        temp = 1./sqrt(2*pi*sigmaM).*exp(-(reshape(img,N,3)-MU).^2./(2*sigmaM));
        temp(temp<0.000001) = 0.000001;%防止出现0
        temp=repmat(temp(:),1,cluster_num);
%       pw(i,:) = log(pc(i)) + log(temp);
        pixpc =  temp*pc';
        rgb_pixpc=reshape(pixpc,N,3);
        pw(:,i)=prod(rgb_pixpc,2);
    end
    pw = pw./(repmat(sum(pw,2),1,cluster_num));%归一
    %----------M---------------------
    %更新参数集
    for i = 1:cluster_num
         pc(i) = mean(pw(:,i));
         avgRgb=pw(:,i)'*reshape(img(:),N,3);
         mu(:,i) = reshape(avgRgb/sum(pw(:,i)),3,1);
         diff=reshape(img,N,3)-repmat(reshape(mu(:,i),1,3),N,1);
         sigma(:,i)=(diff.*diff)'*pw(:,i)/sum(pw(:,i));
    end
    %------------show-result---------------
    [~,label] = max(pw');
    %改大小
    label = reshape(label,size(img,1),size(img,2));
    imshow(label,[])
    title(['iter = ',num2str(iter)]);
    pause(0.1);
    
    % 录制gif
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if iter == 1
        imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    
    iter = iter + 1;
end

