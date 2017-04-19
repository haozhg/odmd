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
% Authors: Hao Zhang, Princeton University
%          haozhang@princeton.edu
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
xq = interp1(t,x,time);
x = xq'; t = time;
% true dynamics, eigenvalues
[n, m] = size(x);
A = zeros(2,2,m);
evals = zeros(2,m);
for k = 1:m
    A(:,:,k) = [0, 1+epsilon*t(k); -(1+epsilon*t(k)),0]; % continuous time dynamics
    evals(:,k) = eig(A(:,:,k)); % analytical continuous time eigenvalues
end


% visualize snapshots
figure, hold on
plot(t,x(1,:),'x-',t,x(2,:),'o-','LineWidth',2)
xlabel('Time','Interpreter','latex')
title('Snapshots','Interpreter','latex')
fl = legend('$x_1(t)$','$x_2(t)$');
set(fl,'Interpreter','latex');
box on
set(gca,'FontSize',20,'LineWidth',2)


% batch DMD
q = 20;
AbatchDMD = zeros(2,2,m);
evalsbatchDMD = zeros(2,m);
tic
for k = q+1:m
    AbatchDMD(:,:,k) = x(:,2:k)*pinv(x(:,1:k-1));
    evalA = eig(AbatchDMD(:,:,k));
    evalsbatchDMD(:,k) = log(evalA)/dt;
end
elapsed_time = toc;
fprintf('Batch DMD, elapsed time: %f seconds\n', elapsed_time)

% Online DMD lambda = 1
q = 20;
AonlineDMD = zeros(2,2,m);
evalsonlineDMD1 = zeros(2,m);
% creat object and initialize with first q snapshot pairs
odmd = OnlineDMD(2,1);
odmd.initialize(x(:,1:q-1),x(:,2:q));
% online DMD
tic
for k = q+1:m
    odmd.update(x(:,k-1),x(:,k));
    AonlineDMD(:,:,k) = odmd.A;
    evalA = eig(AonlineDMD(:,:,k));
    evalsonlineDMD1(:,k) = log(evalA)/dt;
end
elapsed_time = toc;
fprintf('Online DMD, lambda = 1, elapsed time: %f seconds\n', elapsed_time)

% Online DMD, lambda = 0.9
q = 20;
AonlineDMD = zeros(2,2,m);
evalsonlineDMD09 = zeros(2,m);
% creat object and initialize with first q snapshot pairs
odmd = OnlineDMD(2,0.9);
odmd.initialize(x(:,1:q-1),x(:,2:q));
% online DMD
tic
for k = q+1:m
    odmd.update(x(:,k-1),x(:,k));
    AonlineDMD(:,:,k) = odmd.A;
    evalA = eig(AonlineDMD(:,:,k));
    evalsonlineDMD09(:,k) = log(evalA)/dt;
end
elapsed_time = toc;
fprintf('Online DMD, lambda = 0.9, elapsed time: %f seconds\n', elapsed_time)


% visualize imaginary part of the continous time eigenvalues
% from true, batch, online (lambda=1), and online (lambda=0.9)
index = q+1:length(t); tq = t(index);
figure, hold on
plot(t,imag(evals(1,:)),'k-','LineWidth',3)
plot(tq,imag(evalsbatchDMD(1,index)),'-','LineWidth',3)
plot(tq,imag(evalsonlineDMD1(1,index)),'--','LineWidth',3)
plot(tq,imag(evalsonlineDMD09(1,index)),'-','LineWidth',3)
xlabel('Time','Interpreter','latex'), ylabel('Im')
title('Imaginary part of eigenvalues','Interpreter','latex')
fl = legend('True','batch','online, $\lambda=1$','online, $\lambda=0.9$');
set(fl,'Interpreter','latex','Location','northwest');
ylim([1,2]), xlim([0,10])
box on
set(gca,'FontSize',18,'LineWidth',2)