% An example to demonstrate online dynamic mode decomposition
% 
% We take a 2D time varying system given by dx/dt = A(t)x
% where x = [x1,x2]', A(t) = [0,w(t);-w(t),0], 
% w(t)=1+epsilon*t, epsilon=0.1. The slowly time varying eigenvlaues of A(t)
% are pure imaginary, +(1+0.1t)j and -(1+0.1t)j, where j is the imaginary unit.
% 
% At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)], 
% Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs, 
% we would like to compute Ak = Yk*pinv(Xk). This can be done by brute-force 
% batch DMD, and by efficient rank-1 updating online DMD algrithm. Batch DMD 
% computes DMD matrix by brute-force taking the pseudo-inverse directly.
% Online DMD computes the DMD matrix by using efficient rank-1 update idea.
% 
% We compare the performance of online DMD (with weighting=1,0.9) with the 
% brute-force batch DMD approach in terms of tracking time varying eigenvalues, 
% by comparison with the analytical solution. Online DMD (weighting=1) and 
% batch DMD should agree with each other (up to machine round-offer errors).
% 
% Authors: 
%     Hao Zhang
%     Clarence W. Rowley
% 
% References:
% Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta. 
% "Online dynamic mode decomposition for time-varying systems." 
% SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.
%             
% Date created: April 2017

% define dynamics
epsilon = 1e-1;
dyn = @(t,x) ([0, 1+epsilon*t; -(1+epsilon*t),0])*x;
% generate data
dt = 1e-1;
tspan = 0:dt:10;
x0 = [1;0];
[tq,xq] = ode45(dyn, tspan, x0);
% extract snapshot pairs
xq = xq'; tq = tq';
x = xq(:,1:end-1); y = xq(:,2:end); time = tq(2:end);
% true dynamics, eigenvalues
[n, m] = size(x);
A = zeros(n,n,m);
evals = zeros(n,m);
for k = 1:m
    A(:,:,k) = [0, 1+epsilon*time(k); -(1+epsilon*time(k)),0]; % continuous time dynamics
    evals(:,k) = eig(A(:,:,k)); % analytical continuous time eigenvalues
end


% visualize snapshots
figure, hold on
plot(tq,xq(1,:),'x-',tq,xq(2,:),'o-','LineWidth',2)
xlabel('Time','Interpreter','latex')
title('Snapshots','Interpreter','latex')
fl = legend('$x_1(t)$','$x_2(t)$');
set(fl,'Interpreter','latex');
box on
set(gca,'FontSize',20,'LineWidth',2)


% batch DMD
q = 10;
AbatchDMD = zeros(n,n,m);
evalsbatchDMD = zeros(n,m);
tic
for k = q+1:m
    AbatchDMD(:,:,k) = y(:,1:k)*pinv(x(:,1:k));
    evalsbatchDMD(:,k) = log(eig(AbatchDMD(:,:,k)))/dt;
end
elapsed_time = toc;
fprintf('Batch DMD, elapsed time: %f seconds\n', elapsed_time)

% Online DMD weighting = 1
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
fprintf('Online DMD, weighting = 1, elapsed time: %f seconds\n', elapsed_time)

% Online DMD, weighting = 0.9
evalsonlineDMD2 = zeros(n,m);
% creat object and initialize with first q snapshot pairs
odmd = OnlineDMD(n,0.9);
odmd.initialize(x(:,1:q),y(:,1:q));
% online DMD
tic
for k = q+1:m
    odmd.update(x(:,k),y(:,k));
    evalsonlineDMD2(:,k) = log(eig(odmd.A))/dt;
end
elapsed_time = toc;
fprintf('Online DMD, weighting = 0.9, elapsed time: %f seconds\n', elapsed_time)


% visualize imaginary part of the continous time eigenvalues
% from true, batch, online (rho=1), and online (rho=0.9)
updateindex = q+1:m;
figure, hold on
plot(time,imag(evals(1,:)),'k-','LineWidth',2)
plot(time(updateindex),imag(evalsbatchDMD(1,updateindex)),'-','LineWidth',2)
plot(time(updateindex),imag(evalsonlineDMD1(1,updateindex)),'--','LineWidth',2)
plot(time(updateindex),imag(evalsonlineDMD2(1,updateindex)),'--','LineWidth',2)
xlabel('Time','Interpreter','latex'), ylabel('Im($\lambda_{DMD}$)','Interpreter','latex')
fl = legend('True','Batch','Online, $wf=1$','Online, $wf=0.9$');
set(fl,'Interpreter','latex','Location','northwest');
ylim([1,2]), xlim([0,10])
box on
set(gca,'FontSize',20,'LineWidth',2)