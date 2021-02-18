% WindowDMD is a class that implements window dynamic mode decomposition
% The time complexity (multiply-add operation for one iteration) is
% O(8n^2), and space complexity is O(2wn+2n^2), where n is the state
% dimension, w is the window size.
% 
% Algorithm description:
%       At time step k, define two matrix 
%       X(k) = [x(k-w+1),x(k-w+2),...,x(k)], Y(k) = [y(k-w+1),y(k-w+2),
%       ...,y(k)], that contain the recent w snapshot pairs from a finite 
%       time window, where x(k), y(k) are the n dimensional state vector, 
%       y(k) = f(x(k)) is the image of x(k), f() is the dynamics. 
%       Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1))
%       , then x(k), y(k) should be measurements corresponding to 
%       consecutive states z(k-1) and z(k).
%       At time step k+1, we need to forget old snapshot pair xold = 
%       x(k-w+1), yold = y(k-w+1), and remember new snapshot pair xnew = 
%       x(k+1), ynew = y(k+1).
%       We would like to update the DMD matrix Ak = Yk*pinv(Xk) recursively 
%       by efficient rank-2 updating window DMD algorithm.
%       An exponential weighting factor can be used to place more weight on
%       recent data.
%        
% Usage:
%       wdmd = WindowDMD(n,w,weighting)
%       wdmd.initialize(Xw,Yw)
%       wdmd.update(xnew, ynew)
%       [evals, modes] = wdmd.computemodes()
%
% properties:
%       n: state dimension
%       w: finite time window size
%       weighting: weighting factor in (0,1]
%       timestep: number of snapshot pairs processed
%       Xw: recent w snapshots x stored in Xw, size n by w
%       Yw: recent w snapshots y stored in Yw, size n by w
%       A: DMD matrix for w snapshot pairs, size n by n
%       P: Matrix that contains information about recent w snapshots, 
%          size n by n
% 
% methods:
%       initialize(Xw, Yw), initialize window DMD algorithm
%       update(xnew, ynew), update by forgetting old snapshot pairs, 
%       and remeber new snapshot pair, move sliding window forward
%       At time k+1, X(k+1) = [x(k-w+2),x(k-w+2),...,x(k+1)], 
%       Y(k+1) = [y(k-w+2),y(k-w+2),...,y(k+1)], 
%       we should take xnew = x(k+1), ynew = y(k+1)
%       computemodes(), compute and return DMD eigenvalues and DMD modes
%
% Authors: 
%   Hao Zhang
%   Clarence W. Rowley
% 
% Reference:
% Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
% ``Online Dynamic Mode Decomposition for Time-varying Systems", 
% in production, 2017. Available on arXiv.
% 
% Created:
%   April 2017.
%
% To look up the documentation, type help WindowDMD

classdef WindowDMD < handle
    properties
        n = 0;              % state dimension
        w = 0;              % window size
        weighting = 1;      % weighting factor
        timestep = 0;       % number of snapshots processed
        Xw;     % recent w snapshots x stored in matrix Xw
        Yw;     % recent w snapshots y stored in matrix Yw
        A;      % DMD matrix for w snapshot pairs, size n by n
        P;      % Matrix that contains information about recent w snapshots
                % , size n by n
    end
    
    methods
        function obj = WindowDMD(n, w, weighting)
            % Creat an object for window DMD
            % Usage: wdmd = WindowDMD(n,w,weighting)
            if nargin == 3
                obj.n = n;
                obj.w = w;
                obj.weighting = weighting;
                obj.Xw = zeros(n,w);
                obj.Yw = zeros(n,w);
                obj.A = zeros(n,n);
                obj.P = zeros(n,n);
            end
        end
        
        function initialize(obj, Xw, Yw)
            % Initialize WindowDMD with w snapshot pairs stored in (Xw, Yw)
            % Usage: wdmd.initialize(Xw,Yw)
            
            % initialize Xw, Yw
            obj.Xw = Xw; obj.Yw = Yw;
            % initialize A, P
            q = length(Xw(1,:));
            if(obj.timestep == 0 && obj.w == q && rank(Xw) == obj.n)
                weight = (sqrt(obj.weighting)).^(q-1:-1:0);
                Xw = Xw.*weight;
                Yw = Yw.*weight;
                obj.A = Yw*pinv(Xw);
                obj.P = inv(Xw*Xw')/obj.weighting;
            end
            obj.timestep = obj.timestep + q;
        end
        
        function update(obj, xnew, ynew)    
            % Update the DMD computation by sliding the finite time window 
            % forward.
            % Forget the oldest pair of snapshots (xold, yold), and 
            % remembers the newest pair of snapshots (xnew, ynew) in the 
            % new time window. If the new finite time window at time step 
            % k+1 includes recent w snapshot pairs as
            % X(k+1) = [x(k-w+2),x(k-w+3),...,x(k+1)], 
            % Y(k+1) = [y(k-w+2),y(k-w+3),...,y(k+1)],
            % where y(k) = f(x(k)) and f is the dynamics, then we should 
            % take xnew = x(k+1), ynew = y(k+1)
            % Usage: wdmd.update(xnew, ynew)
            
            % define old snapshots to be discarded
            xold = obj.Xw(:,1); yold = obj.Yw(:,1);
            % Update recent w snapshots
            obj.Xw = [obj.Xw(:,2:end), xnew];
            obj.Yw = [obj.Yw(:,2:end), ynew];
            
            % direct rank-2 update
            % define matrices
            U = [xold, xnew]; V = [yold, ynew]; 
            C = diag([-(obj.weighting)^(obj.w),1]);
            % compute PkU matrix vector product beforehand
            PkU = obj.P*U;
            % compute AkU matrix vector product beforehand
            AkU = obj.A*U;
            % compute Gamma
            Gamma = inv(inv(C)+U'*PkU);
            % update A
            obj.A = obj.A + (V-AkU)*(Gamma*PkU');
            % update P
            obj.P = (obj.P - PkU*(Gamma*PkU'))/obj.weighting;
            % ensure P is SPD by taking its symmetric part
            obj.P = (obj.P+(obj.P)')/2;
            
            % time step + 1
            obj.timestep = obj.timestep + 1;
        end
        
        function [evals, modes] = computemodes(obj)
            % Compute DMD eigenvalues and DMD modes at current time step
            % Usage: [evals, modes] = wdmd.computemodes()
            [modes, evals] = eig(obj.A, 'vector');
        end
    end
end