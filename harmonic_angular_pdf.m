% script generating an angular distribution from the first harmonic mode of
% the prerturbation (with amplitude eps) to the uniform distribution.

function [angle,cdf]=harmonic_angular_pdf(eps_vec,n_particles);
phi=0*pi;

%eps=-0.97; % set eps in the [-1,1] range

x = 0:2*pi/n_particles:2*pi;
pdf = 1 + eps_vec.*cos((x-phi).*2);
pdf = pdf / sum(pdf);
cdf = cumsum(pdf);
projection = spline(cdf, x, rand(1,n_particles));

angle=projection; % array of angles
aa=isnan(cdf);
index=find(aa==1)
if(index==1)
    disp('nan found')
end
%figure;hist(th,51)
end