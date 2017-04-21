% OnlineDMD is a class that implements online dynamic mode decomposition
% The time complexity (for one iteration) is O(n^2), and space complexity is 
% O(n^2), where n is the state dimension.
%
% Algorithm description:
%       At time step k, define two matrix Xk = [x(1),x(2),...,x(k)], Yk = [y(1),y(2),...,y(k)],
%       that contain all the past snapshot pairs, where x(k), y(k) are the n 
%       dimensional state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics. 
%       Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k)
%       should be measurements corresponding to consecutive states z(k-1) and z(k).
%       We would like to update the DMD matrix Ak = Yk*pinv(Xk) recursively 
%       by efficient rank-1 updating online DMD algorithm.
%
% Usage:
%       odmd = OnlineDMD(n,lambda)
%       odmd.initialize(Xq,Yq)
%       odmd.initilizeghost()
%       odmd.update(x,y)
%       [evals, modes] = odmd.computemodes()
%        
% properties:
%       n: state dimension
%       lambda: weighting factor between 0 and 1
%       timestep: number of snapshot pairs processed
%       A: DMD matrix, size n by n
%       P: matrix that contains information about past snapshots, size n by n
% 
% methods:
%       initialize(Xq, Yq), initialize online DMD algorithm with q snapshot pairs stored in (Xq, Yq)
%       initializeghost(), initialize online DMD algorithm with epsilon small (1e-15) ghost snapshot pairs before t=0
%       update(x,y), update when new snapshot pair (x,y) becomes available
%                   Here, if the (discrete-time) dynamics are given by 
%                   z(k) = f(z(k-1)), then (x,y) should be measurements 
%                   correponding to consecutive states z(k-1) and z(k).
%       computemodes(), compute and return DMD eigenvalues and DMD mdoes
%
% Authors: 
%   Hao Zhang
%   Clarence W. Rowley
% 
% Created:
%   April 2017.
%
% To look up the documentation in the command window, type help OnlineDMD

classdef OnlineDMD < handle
    properties
        n = 0;                      % state dimension
        lambda = 1;                 % weighting factor
        timestep = 0;               % number of snapshots processed
        A;          % DMD matrix
        P;          % matrix that contains information about past snapshots
    end
    
    methods
        function obj = OnlineDMD(n,lambda)
            % Creat an object for online DMD
            % Usage: odmd = OnlineDMD(n,lambda)
            if nargin == 2
                obj.n = n;
                obj.lambda = lambda;
                obj.A = zeros(n,n);
                obj.P = zeros(n,n);
            end
        end
        
        function initialize(obj, Xq, Yq)
            % Initialize OnlineDMD with q snapshot pairs stored in (Xq, Yq)
            % Usage: odmd.initialize(Xq,Yq)
            q = length(Xq(1,:));
            if(obj.timestep == 0 && q>=obj.n)
                sqrtlambda = sqrt(obj.lambda);
                for i = 1:q
                    Xq(:,i) = Xq(:,i)*sqrtlambda^(q-i);
                    Yq(:,i) = Yq(:,i)*sqrtlambda^(q-i);
                end
                obj.A = Yq*pinv(Xq);
                obj.P = inv(Xq*Xq')/obj.lambda;
            end
            obj.timestep = obj.timestep + q;
        end
        
        function initializeghost(obj)
            % Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs before t=0
            % Usage: odmd.initilizeghost()
            epsilon = 1e-15;
            alpha = 1.0/epsilon;
            obj.A = randn(obj.n, obj.n);
            obj.P = alpha*eye(obj.n);
        end
        
        function update(obj, x, y)
            % Update the DMD computation with a new pair of snapshots (x,y)
            % Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then (x,y)
            % should be measurements correponding to consecutive states z(k-1) and z(k).
            % Usage: odmd.update(x, y)
            
            % Compute gamma
            gamma = 1/(1+x'*(obj.P*x));
            % Update A
            obj.A = obj.A + gamma*((y-obj.A*x)*(x'*obj.P));
            % Update P
            obj.P = (obj.P - gamma*((obj.P*x)*(x'*obj.P)))/obj.lambda;
            
            obj.timestep = obj.timestep + 1;
        end
        
        function [evals, modes] = computemodes(obj)
            % Compute and return DMD eigenvalues and DMD modes at current time
            % Usage: [evals, modes] = odmd.modes()
            [modes, evals] = eig(obj.A, 'vector');
        end
    end
end