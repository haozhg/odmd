
%% clean workspace
clear all
close all
clc

%% define dynamics
epsilon = 1e-1;
dyn = @(t,x) ([0, 1+epsilon*t; -(1+epsilon*t),0])*x;

% generate data
dt = 1e-1;
tspan = 0:dt:10;
x0 = [1;0];
[tq,xq] = ode45(dyn, tspan, x0);
% extract snapshot pairs
xq = xq'; tq = tq';
x = xq(:,1:end-1); y = xq(:,2:end); t = tq(2:end);

% true dynamics, eigenvalues and modes
[n, m] = size(x);
A = zeros(2,2,m);
evals = zeros(2,m);
for k = 1:m
    A(:,:,k) = [0, 1+epsilon*t(k); -(1+epsilon*t(k)),0]; % continuous time dynamics
    evals(:,k) = eig(A(:,:,k));
end

%% visualize snapshots
figure
plot(tq,xq(1,:),'k-',tq,xq(2,:),'k--','MarkerSize',12,'LineWidth',1.5)
xlabel('Time $t$','Interpreter','latex')
ylabel('$x_1(t), x_2(t)$','Interpreter','latex')
fl = legend('$x_1$', '$x_2$');
set(fl,'Interpreter','latex','Box','off','FontSize',20,'Location','best');
box on
set(gca,'FontSize',20,'LineWidth',1)

%% run and compare different DMD algorithms
w = 10;

% batch DMD
AbatchDMD = zeros(2,2,m);
evalsbatchDMD = zeros(2,m);
for k = 1:w
    AbatchDMD(:,:,k) = expm(A(:,:,k)*dt);
end
tic
% start at w+1
for k = w+1:m
    AbatchDMD(:,:,k) = y(:,1:k)*pinv(x(:,1:k));
end
batchDMDelapsedTime = toc
for k = 1:m
    evalA = eig(AbatchDMD(:,:,k));
    evalsbatchDMD(:,k) = log(evalA)/dt;
end

% online DMD, rho = 1
AonlineDMD = zeros(2,2,m);
evalsonlineDMD = zeros(2,m);
for k = 1:w
    AonlineDMD(:,:,k) = expm(A(:,:,k)*dt);
end
% initial condition
odmd = OnlineDMD(2,1);
odmd.initialize(x(:,1:w),y(:,1:w));
tic
% start at w+1
for k = w+1:m
    odmd.update(x(:,k),y(:,k));
    AonlineDMD(:,:,k) = odmd.A;
end
onlineDMDelapsedTime = toc
for k = 1:m
    evalA = eig(AonlineDMD(:,:,k));
    evalsonlineDMD(:,k) = log(evalA)/dt;
end
evalsonlineDMD1 = evalsonlineDMD;

% online DMD, rho = 0.95
AonlineDMD = zeros(2,2,m);
evalsonlineDMD = zeros(2,m);
for k = 1:w
    AonlineDMD(:,:,k) = expm(A(:,:,k)*dt);
end
% initial condition
odmd = OnlineDMD(2,0.95);
odmd.initialize(x(:,1:w),y(:,1:w));
tic
% start at w+1
for k = w+1:m
    odmd.update(x(:,k),y(:,k));
    AonlineDMD(:,:,k) = odmd.A;
end
onlineDMDelapsedTime = toc
for k = 1:m
    evalA = eig(AonlineDMD(:,:,k));
    evalsonlineDMD(:,k) = log(evalA)/dt;
end
evalsonlineDMD2 = evalsonlineDMD;

% online DMD, rho = 0.8
AonlineDMD = zeros(2,2,m);
evalsonlineDMD = zeros(2,m);
for k = 1:w
    AonlineDMD(:,:,k) = expm(A(:,:,k)*dt);
end
% initial condition
odmd = OnlineDMD(2,0.8);
odmd.initialize(x(:,1:w),y(:,1:w));
tic
% start at w+1
for k = w+1:m
    odmd.update(x(:,k),y(:,k));
    AonlineDMD(:,:,k) = odmd.A;
end
onlineDMDelapsedTime = toc
for k = 1:m
    evalA = eig(AonlineDMD(:,:,k));
    evalsonlineDMD(:,k) = log(evalA)/dt;
end
evalsonlineDMD3 = evalsonlineDMD;

% streaming DMD
evalsstreamingDMD = zeros(2,m);
% initial
evalsstreamingDMD(:,1) = evals(:,1);
% streaming DMD
max_rank = 2;
sdmd = StreamingDMD(max_rank);
for k = 2:m
    sdmd = sdmd.update(x(:,k), y(:,k));
    [~, evalA] = sdmd.compute_modes();
    evalsstreamingDMD(:,k) = log(evalA)/dt;
end

% mini-batch DMD
AminibatchDMD = zeros(2,2,m);
evalsminibatchDMD = zeros(2,m);
for k = 1:w
    AminibatchDMD(:,:,k) = expm(A(:,:,k)*dt);
end
tic
% start at w+1
for k = w+1:m
    AminibatchDMD(:,:,k) = y(:,k-w+1:k)*pinv(x(:,k-w+1:k));
end
minibatchDMDelapsedTime = toc
for k = 1:m
    evalA = eig(AminibatchDMD(:,:,k));
    evalsminibatchDMD(:,k) = log(evalA)/dt;
end

% window DMD
AwindowDMD = zeros(2,2,m);
evalswindowDMD = zeros(2,m);
for k = 1:w
    AwindowDMD(:,:,k) = expm(A(:,:,k)*dt);
end
% initialization
wdmd = WindowDMD(2,w,1);
wdmd.initialize(x(:,1:w), y(:,1:w));
tic
% start at w+1
for k = w+1:m
    wdmd.update(x(:,k), y(:,k));
    AwindowDMD(:,:,k) = wdmd.A;
end
windowDMDelapsedTime = toc
for k = 1:m
    evalA = eig(AwindowDMD(:,:,k));
    evalswindowDMD(:,k) = log(evalA)/dt;
end

%% Compare various DMD algorithm
index = w+1:length(t);

% plot Imaginary part

figure, hold on
plot(t,imag(evals(1,:)),'k-','LineWidth',1)
plot(t(1,index),imag(evalsbatchDMD(1,index)),'s-','LineWidth',1,'MarkerSize',12,'MarkerIndices',1:10:length(index))
plot(t(1,index),imag(evalsminibatchDMD(1,index)),'o-','LineWidth',1,'MarkerSize',12,'MarkerIndices',1:10:length(index))
plot(t(1,index),imag(evalsstreamingDMD(1,index)),'x-','LineWidth',1,'MarkerSize',12,'MarkerIndices',7:10:length(index))
plot(t(1,index),imag(evalsonlineDMD1(1,index)),'>-','LineWidth',1,'MarkerSize',12,'MarkerIndices',4:10:length(index))
plot(t(1,index),imag(evalsonlineDMD2(1,index)),'*-','LineWidth',1,'MarkerSize',12,'MarkerIndices',1:10:length(index))
plot(t(1,index),imag(evalsonlineDMD3(1,index)),'d-','LineWidth',1,'MarkerSize',12,'MarkerIndices',4:10:length(index))
plot(t(1,index),imag(evalswindowDMD(1,index)),'+-','LineWidth',1,'MarkerSize',12,'MarkerIndices',7:10:length(index))
xlabel('Time $t$','Interpreter','latex'), ylabel('Im$(\lambda)$','Interpreter','latex')
fl = legend('true','batch','mini-batch','streaming','online, $\rho=1$','online, $\rho=0.95$','online, $\rho=0.8$','window');
set(fl,'Interpreter','latex','Location','northwest','FontSize',20,'Box','off');
xlim([0,10]), ylim([1,2])

box on
set(gca,'FontSize',20,'LineWidth',1)
