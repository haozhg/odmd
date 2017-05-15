% An example to demonstrate online dynamic mode decomposition
% 
% We take a 2D time varying system given by dx/dt = A(t)x
% where x = [x1,x2]', A(t) = [0,w(t);-w(t),0], 
% w(t)=1+epsilon*t, epsilon=0.1. The slowly time varying eigenvlaues of A(t)
% are pure imaginary, i.e, +(1+0.1t)j and -(1+0.1t)j, where j is the imaginary unit
% 
% At time step k, define two matrix Xk = [x(1),x(2),...,x(k)], Yk = [y(1),y(2),...,y(k)],
% that contain all the past snapshot pairs, we would like to compute 
% Ak = Yk*pinv(Xk). This can be done by brute-force batch DMD, 
% and by efficient rank-1 updating online DMD algrithm.
% 
% Batch DMD computes DMD matrix by brute-force taking the pseudo-inverse directly
% 
% Online DMD computes the DMD matrix by using efficient rank-1 update idea
% 
% We compare the performance of online DMD (with lambda=1,0.9) with the brute-force batch DMD
% approach in terms of tracking time varying eigenvalues, by comparison with the analytical solution
% 
% Authors: 
%   Hao Zhang
%   Clarence W. Rowley
% 
% Reference:
% Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
% ``Online Dynamic Mode Decomposition for Time-varying Systems", 
% in production, 2017. To be submitted for publication, available on arXiv.
% 
% Date created: April 2017

% define dynamics
epsilon = 1e-1;
dyn = @(t,x) ([0, 1+epsilon*t; -(1+epsilon*t),0])*x;
% generate data
tspan = [0 10];
x0 = [1;0];
[t,x] = ode45(dyn, tspan, x0);
% interpolate uniform time step
dt = 1e-1;
time = 0:dt:max(tspan);
xq = interp1(t,x,time); xq = xq';
% extract snapshot pairs
x = xq(:,1:end-1); y = xq(:,2:end); t = time(2:end);
% true dynamics, eigenvalues
[n, m] = size(x);
A = zeros(n,n,m);
evals = zeros(n,m);
for k = 1:m
    A(:,:,k) = [0, 1+epsilon*t(k); -(1+epsilon*t(k)),0]; % continuous time dynamics
    evals(:,k) = eig(A(:,:,k)); % analytical continuous time eigenvalues
end


% visualize snapshots
figure, hold on
plot(time,xq(1,:),'x-',time,xq(2,:),'o-','LineWidth',2)
xlabel('Time','Interpreter','latex')
title('Snapshots','Interpreter','latex')
fl = legend('$x_1(t)$','$x_2(t)$');
set(fl,'Interpreter','latex');
box on
set(gca,'FontSize',20,'LineWidth',2)


% batch DMD
q = 20;
AbatchDMD = zeros(n,n,m);
evalsbatchDMD = zeros(n,m);
tic
for k = q+1:m
    AbatchDMD(:,:,k) = y(:,1:k)*pinv(x(:,1:k));
    evalsbatchDMD(:,k) = log(eig(AbatchDMD(:,:,k)))/dt;
end
elapsed_time = toc;
fprintf('Batch DMD, elapsed time: %f seconds\n', elapsed_time)

% Online DMD lambda = 1
q = 20;
evalsonlineDMD1 = zeros(n,m);
% creat object and initialize with first q snapshot pairs
odmd = OnlineDMD(n,1);
odmd.initialize(x(:,1:q),y(:,1:q));
% online DMD
tic
for k = q+1:m
    odmd.update(x(:,k),y(:,k));
    evalsonlineDMD1(:,k) = log(eig(odmd.A))/dt;
end
elapsed_time = toc;
fprintf('Online DMD, lambda = 1, elapsed time: %f seconds\n', elapsed_time)

% Online DMD, lambda = 0.9
q = 20;
evalsonlineDMD09 = zeros(n,m);
% creat object and initialize with first q snapshot pairs
odmd = OnlineDMD(n,0.9);
odmd.initialize(x(:,1:q),y(:,1:q));
% online DMD
tic
for k = q+1:m
    odmd.update(x(:,k),y(:,k));
    evalsonlineDMD09(:,k) = log(eig(odmd.A))/dt;
end
elapsed_time = toc;
fprintf('Online DMD, lambda = 0.9, elapsed time: %f seconds\n', elapsed_time)


% visualize imaginary part of the continous time eigenvalues
% from true, batch, online (lambda=1), and online (lambda=0.9)
updateindex = q+1:m;
figure, hold on
plot(t,imag(evals(1,:)),'k-','LineWidth',3)
plot(t(updateindex),imag(evalsbatchDMD(1,updateindex)),'-','LineWidth',3)
plot(t(updateindex),imag(evalsonlineDMD1(1,updateindex)),'--','LineWidth',3)
plot(t(updateindex),imag(evalsonlineDMD09(1,updateindex)),'-','LineWidth',3)
xlabel('Time','Interpreter','latex'), ylabel('Im')
title('Imaginary part of eigenvalues','Interpreter','latex')
fl = legend('True','batch','online, $\lambda=1$','online, $\lambda=0.9$');
set(fl,'Interpreter','latex','Location','northwest');
ylim([1,2]), xlim([0,10])
box on
set(gca,'FontSize',18,'LineWidth',2)