%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Program to fit the fluorescent intensity signal 
%to a sinusoidal curve. The total contribution of the
%signal is the obtained by direct summation taking into
%account the orientation of individual cells. 
%Jorge Arrieta Nov-2024
%the program now calls the harmonic 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all

addpath('./sine_fit/');
    
load fluo_single_cell.mat;%intensity from single cell is loaded
[Theta,I]=sort(Theta,'ascend');
Theta=Theta*pi/180;
Fluo_cell=Fluo_cell(I);
[param, err_1]=sine_fit(Theta,Fluo_cell);
Fluo_fit=param(1) + param(2) * sin( param(3) + 2*pi*param(4)*Theta);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Let us calculate the total intensity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=80000;
Theta_1=rand(N,1)*2*pi;
n_angles=40;
eps_angle=[-0.99:1.98/n_angles:0.99];

Theta_2=zeros(N,n_angles);

%Fluo_unif=param(1) + iiparam(2) * sin( param(3) + 2*pi*param(4)*Theta_1);
%avg_I_unif=sum(Fluo_unif)/(max(Fluo_fit)*N);


for ii=1:length(eps_angle);
    %Theta_2(:,ii)=normrnd(angle(ii),0.1,[N,1]);
    [intensity,A]=harmonic_angular_pdf(eps_angle(ii),N);
    Theta_2(:,ii)=intensity;
    cdf(:,ii)=A;

    Fluo_gauss(:,ii)=param(1) + param(2) * sin( param(3) + 2*pi*param(4)*Theta_2(:,ii));
    avg_I_gauss(ii)=sum(Fluo_gauss(:,ii));
    angulo_medio(ii)=mean(mod(Theta_2(:,ii),pi));
end