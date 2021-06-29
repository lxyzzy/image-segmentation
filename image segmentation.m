clear all;close all;clc;
img = double(rgb2gray(imread('lena.jpg')));
%���þ�����
cluster_num = 5;
% ���ó�ʼ��ֵ�ͷ����ֵΪ����ڸ�ɫ����ѡ�㲢ȡ������
mu=[50 70 100 190 220];
sigma = 15^2*ones(1,cluster_num);
pw = zeros(cluster_num,size(img,1)*size(img,2));
pc = rand(1,cluster_num);
pc = pc/sum(pc);%������ʹ�һ��
max_iter = 50;%�Ե�����������Ϊֹͣ������
iter = 1;
while iter <= max_iter
    %----------E-------------------
    for i = 1:cluster_num
        MU = repmat(mu(i),size(img,1)*size(img,2),1);
        %��˹ģ��
        temp = 1/sqrt(2*pi*sigma(i))*exp(-(img(:)-MU).^2/2/sigma(i));
        temp(temp<0.000001) = 0.000001;%��ֹ����0
        pw(i,:) = pc(i) * temp;
    end
    pw = pw./(repmat(sum(pw),cluster_num,1));%��һ
    %----------M---------------------
    %���²�����
    for i = 1:cluster_num
         pc(i) = mean(pw(i,:));
         mu(i) = pw(i,:)*img(:)/sum(pw(i,:));
         sigma(i) = pw(i,:)*((img(:)-mu(i)).^2)/sum(pw(i,:));
    end
    %------------show-result---------------
    [~,label] = max(pw);
    %�Ĵ�С
    label = reshape(label,size(img));
    imshow(label,[])
    title(['iter = ',num2str(iter)]);
    pause(0.1);
    M(iter,:) = mu;
    S(iter,:) = sigma;
    
    % ¼��gif
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
%����ֵ�뷽��ĵ���������ʾ����
figure
for i = 1:cluster_num
    plot(M(:,i));
    hold on
end
title('��ֵ�仯����');
figure
for i = 1:cluster_num
    plot(S(:,i));
    hold on
end
title('����仯����');
�ڶ�����
clear all;close all;clc;
img = double(imread('lena.jpg'));
N=size(img,1)*size(img,2);
%���þ�����
cluster_num = 5;
% muΪ�������RGB��ͨ����ֵ��ÿ��Ϊһ������ľ�ֵ��
% ��ʼֵ����ͼ�лơ��̡��졢�ס��ϲ������ȡһ��õ���RGBֵ
mu=[255,115,216,247,75;...
        220,124,50,215,45;...
        50,31,39,185,75];
% sigmaΪ��������ͨ���ķ����ʼ�������Ϊ20^2
sigma=20^2*ones(3,cluster_num);
pw = zeros(N,cluster_num); %�����ص����ڸ�����ĸ��ʣ������Ҫ�Ծ����һ��
pc = rand(1,cluster_num);% ����������ĸ��ʷֲ�
pc = pc/sum(pc);%������ʹ�һ��
max_iter = 30;%�Ե�����������Ϊֹͣ������
iter = 1;
while iter <= max_iter
    %----------E-------------------
    for i = 1:cluster_num
        %�������--�ٶȿ�
        MU = repmat(reshape(mu(:,i),1,3),N,1);
        %��˹ģ��
        sigmaM=repmat(reshape(sigma(:,i),1,3),N,1);
        temp = 1./sqrt(2*pi*sigmaM).*exp(-(reshape(img,N,3)-MU).^2./(2*sigmaM));
        temp(temp<0.000001) = 0.000001;%��ֹ����0
        temp=repmat(temp(:),1,cluster_num);
%       pw(i,:) = log(pc(i)) + log(temp);
        pixpc =  temp*pc';
        rgb_pixpc=reshape(pixpc,N,3);
        pw(:,i)=prod(rgb_pixpc,2);
    end
    pw = pw./(repmat(sum(pw,2),1,cluster_num));%��һ
    %----------M---------------------
    %���²�����
    for i = 1:cluster_num
         pc(i) = mean(pw(:,i));
         avgRgb=pw(:,i)'*reshape(img(:),N,3);
         mu(:,i) = reshape(avgRgb/sum(pw(:,i)),3,1);
         diff=reshape(img,N,3)-repmat(reshape(mu(:,i),1,3),N,1);
         sigma(:,i)=(diff.*diff)'*pw(:,i)/sum(pw(:,i));
    end
    %------------show-result---------------
    [~,label] = max(pw');
    %�Ĵ�С
    label = reshape(label,size(img,1),size(img,2));
    imshow(label,[])
    title(['iter = ',num2str(iter)]);
    pause(0.1);
    
    % ¼��gif
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

