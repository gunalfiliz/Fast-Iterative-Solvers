clear all
close all
clc

%filename = 'gmres_test_msr.txt';
filename = 'cg_test_msr.txt';
max_iterations = 800;  % Maximum number of iterations
%Read the file
[matrix_dim, array_size, JM, VM, symmetry] = Read_MSR_File(filename);

% Helpers
x = ones(matrix_dim,1);
x0 =zeros(matrix_dim,1);
tol = 1e-8;

b = MSR_vector_multiplication(JM, VM, x, matrix_dim,symmetry); %b = Ax
r0 = b - MSR_vector_multiplication(JM, VM, x0, matrix_dim,symmetry);

% For non-symmetric matrices, we use GMRES method
if (symmetry == ' n') %GMRES
    m = 600; % restart parameter
    V = zeros(matrix_dim,m+1);
    V(:,1) = r0/((iprod(r0,r0))^0.5);
    g = zeros(m+1,1);
    g(1)= ((iprod(r0,r0))^0.5);
    H= zeros(m);
    GMRES_figures(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, r0)

    % Runtime
    [restarts,krylov,time] = runtime(JM,VM,V,H,matrix_dim,symmetry,x0,g,m,tol,r0);
end
% Apply Conjugate Gradient Method if the Given Matrix is Symmetric
if (symmetry == ' s') 
    [xm,r2,norma,m] = Conjugate_Gradient(JM,VM,matrix_dim,symmetry,x0,r0,tol);
    semilogy(abs(r2))
    hold on
    % Check the orthogonality of the Krylov vectors
    semilogy(abs(norma))
    xlabel('Number of Iterations');
    ylabel('Norm')
    legend('2-Norm','A-Norm')
end
%% Some Test Cases

% Test Case 0
% to test if the matrix vector production works fine
% JM_test=[6,8,9,9,10,2,3,4,3];
% VM_test=[1,2,1,2,0,5,4,3,4];
% vektor_test=[1 4 5 3];
% dim_test=4;
% x_test = ones(dim_test,1);
% x0_test =zeros(dim_test,1);

%[y_test, yd_test, ya0_test] = MSR_vector_multiplication(JM_test, VM_test, vektor_test, dim_test,symmetry);
% b_test = MSR_vector_multiplication(JM_test, VM_test, x_test, dim_test,symmetry); %b = Ax
% r0_test = b - MSR_vector_multiplication(JM_test, VM_test, x0_test, dim_test,symmetry);


% Test Case 1
% test if matrix vector production works good with the given matrix
% vektor = ones(1030,1);
% [y, yd, ya0] = MSR_vector_multiplication(JM, VM, vektor, matrix_dim,symmetry);

%% Functions

function [matrix_dim, array_size, JM, VM, symmetry] = Read_MSR_File(filename)
% Open the file
fileID = fopen(filename, 'r');

% Read the first line (symmetry information)
symmetry = fgetl(fileID);

% Read the second line (matrix dimension and array size)
line = fgetl(fileID);
values = sscanf(line, '%d');
matrix_dim = values(1);
array_size = values(2);

% Initialize arrays to store JM and VM values
JM = zeros(array_size, 1);
VM = zeros(array_size, 1);

% Read the remaining lines (JM and VM values)
for i = 1:array_size
    line = fgetl(fileID);
    values = sscanf(line, '%d %f');
    JM(i) = values(1);
    VM(i) = values(2);
end

% Close the file
fclose(fileID);

% Display the read data
disp(['Symmetry: ' symmetry]);
disp(['Matrix Dimension: ' num2str(matrix_dim)]);
disp(['Array Size: ' num2str(array_size)]);
end

% Perform Matrix-Vector Multiplication for MSR format
function [y, yd, ya0]=MSR_vector_multiplication(JM,VM,x,matrix_dim,symmetry)
y=zeros(matrix_dim,1);
yd=zeros(matrix_dim,1);
ya0=zeros(matrix_dim,1);
if symmetry==' n'
    for i=1:1:matrix_dim
        yd(i)=VM(i).*x(i);
        if (JM(i)==length(VM) || JM(i)<=length(VM)) && JM(i)~=JM(i+1)
            for j=JM(i):JM(i+1)-1
                ya0(i) = ya0(i) + dot(VM(j),x(JM(j)));
            end
        end
    end
    y=yd+ya0;
end
if symmetry== ' s'
    for i=1:1:matrix_dim
        yd(i)=VM(i).*x(i);
        if (JM(i)==length(VM) || JM(i)<=length(VM)) && JM(i)~=JM(i+1)
            for j=JM(i):JM(i+1)-1
                if i>JM(j)
                    ya0(i) = ya0(i) + dot(VM(j),x(JM(j)));
                    ya0(JM(j)) = ya0(JM(j)) + dot(VM(j),x(i));
                end
            end
        end
    end
    y=yd+ya0;
end
end

% Perform Inner Product
function y = iprod(a,b)
y=0;
n= length(a);
for k=1:n
    y=y+a(k)*b(k);
end
end

function [x,res,ort,rho]=GMRES(JM, VM, V,H,matrix_dim,symmetry,x,g,m,tol,pre,pcase,r0,restart)
% Check if preconditioning is applied
if(pre==1)
    r0=Preconditioning(JM, VM, matrix_dim,r0,pcase);
    V(:,1) = r0/((iprod(r0,r0))^0.5);
    g = zeros(m+1,1);
    g(1)= ((iprod(r0,r0))^0.5);
end
% Check if restart is selected
if (restart ~= 0)
    g(1)= restart;
    restart = 1;
end

normr0=g(1);
res(1)=g(1)/normr0;
j=1;

% For Full GMRES
while (res(j)>tol) 
    j1=j;
    [v_jp1,hj] = getKrylov(JM,VM,V,matrix_dim,symmetry,j1,pre,pcase);
    V(:,j+1) = v_jp1;
    ort(j)=iprod(V(:,j+1),V(:,1));

    % Hessenberg Matrix
    H(1:size(hj),j) = hj;
    for k=2:j
        h1=c(k-1)*H(k-1,j)+s(k-1)*H(k,j);
        h2=-s(k-1)*H(k-1,j)+c(k-1)*H(k,j);
        H(k-1,j) =h1;
        H(k,j) = h2;
    end
    hq = ((H(j,j)^2+H(j+1,j)^2)^0.5);
    c(j)=H(j,j)/hq;
    s(j)=H(j+1,j)/hq;
    H(j,j)=c(j)*H(j,j)+s(j)*H(j+1,j);
    g(j+1)=-s(j)*g(j);
    rho=abs(g(j+1));
    res(j+1)=rho/normr0;
    if(res(j+1)<=tol || (restart==1 && j==m))
        y=zeros(j,1);
        y(j)=g(j)/H(j,j);
        for i=j-1:-1:1
            for k=j:-1:i+1
                g(i)= g(i)-H(i,k)*y(k);
            end
            y(i)=g(i)/H(i,i);
        end
        for i=1:1:j
            x=x+y(i)*V(j,i);
        end
        break
    end
    g(j)=c(j)*g(j);
    j=j+1;
end
end


function [v_jp1,hj] = getKrylov(JM,VM,V,matrix_dim,symmetry,j,pre,pcase)
w=MSR_vector_multiplication(JM, VM,V(:,j),matrix_dim,symmetry);
if(pre==1)
    w=Preconditioning(JM, VM,matrix_dim,w,pcase);
end
helper_size=size(V(:,j),1);
h=zeros(helper_size,1);
for i=1:1:j
    h_ij=iprod(w,V(:,i));
    w=w-h_ij*V(:,i);
    h(i)=h_ij;
end
h(j+1)=sqrt(iprod(w,w));
hj=h;
v_jp1 =w/h(j+1);
end


function z=Preconditioning(JM,VM,matrix_dim,x,pcase)
size_x=size(x,1);
z=zeros(size_x,1);
%Jacobi
if(pcase==0)
    for i=1:1:matrix_dim
        z(i)=x(i)/VM(i);
    end
end
%Gauss-Seidel
if(pcase==1)
    for i=1:1:matrix_dim
        if (JM(i)==length(VM) || JM(i)<=length(VM)) && JM(i)~=JM(i+1)
            for j=JM(i):JM(i+1)-1
                if  i>JM(j)
                    z(i)=x(i)/VM(j);
                end
                x(i)=x(i)-VM(j)*z(JM(j));
            end
        end
        z(i)=x(i)/VM(i);
    end
end
if(pcase==2)
    % Apply ILU(0) preconditioning
    [PV, PJM]=ilu0_in_msr(JM, VM, matrix_dim);

    % Apply forward and backward substitution
    z=ilu0_forward_backward_substitution(PV, PJM, x, matrix_dim,x);
end
end


% Conjugate Gradient
function [x_m,r2,norma,m]=Conjugate_Gradient(JM,VM,matrix_dim,symmetry,x0,r0,tol)
m=1;
r_m=r0;
p_m=r0;
x_m=x0;
x=ones(matrix_dim,1);
r2(m)=(iprod(r_m,r_m))^.5;

while(r2(m)/r2(1)>tol)
    A=MSR_vector_multiplication(JM,VM,p_m,matrix_dim,symmetry);
    A_pm=iprod(A,p_m);
    alfa=r2(m)^2/A_pm;
    x_m=x_m+alfa*p_m;
    r_m=r_m-alfa*A;
    r2(m+1)=(iprod(r_m,r_m))^0.5;
    beta=r2(m+1)^2/r2(m)^2;
    p_m=r_m+beta*p_m;
    e_k=x_m-x;
    helper_prod=MSR_vector_multiplication(JM,VM,e_k,matrix_dim,symmetry);
    norma(m)=(iprod(helper_prod,e_k))^0.5;
    m=m+1;
end
end

function [restarts, krylov, time] = runtime(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, r0)
    m_values = [10, 30, 50, 100, 200];
    num_m_values = numel(m_values);
    num_methods = 3;
    
    time = zeros(num_methods, num_m_values);
    restarts = zeros(num_methods, num_m_values);
    krylov = repmat(m, num_methods, num_m_values);
    
    for i = 1:num_m_values
        m = m_values(i);
        
        tic
        [~, res, ~, ~] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 0, 0, r0, 0);
        time(1, i) = toc;
        
        tic
        [~, res, ~, ~] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 0, r0, 0);
        time(2, i) = toc;
        
        tic
        [~, res, ~, ~] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 1, r0, 0);
        time(3, i) = toc;

           tic
        [~, res, ~, ~] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1,2, r0, 0);
        time(4, i) = toc;
        
        tic
        [~, ~, rn] = GMRES_restart(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 0, 0, r0);
        time(5, i) = toc;
        restarts(1, i) = rn;
        
        tic
        [~, ~, rj] = GMRES_restart(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 0, r0);
        time(6, i) = toc;
        restarts(2, i) = rj;
        
        tic
        [~, ~, rg] = GMRES_restart(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 1, r0);
        time(7, i) = toc;
        restarts(3, i) = rg;

        tic
        [~, ~, rk] = GMRES_restart(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 2, r0);
        time(8, i) = toc;
        restarts(4, i) = rk;
    end
end


function [x,i,kr]=GMRES_restart(JM,VM,V,H,matrix_dim,symmetry,x0,g,m,tol,pre,pcase,r0)
    rho0=(iprod(r0,r0))^.5;
    rho1=1;
    i=1;
    x=x0;
    kr=0;
    while(rho1>tol)
        [xm,res,ort,rho]=GMRES(JM,VM,V,H,matrix_dim,symmetry,x,g,m,tol,pre,pcase,r0,rho1);
        x=xm;
        rho1=rho/rho0;
        kr=kr+size(ort,2);
        i=i+1;
    end
end

% Applied ILU(0) Algotihm
function [PV, PJM] = ilu0_in_msr(JM, VM, matrix_dim)
    JR=zeros(matrix_dim,1);
    JD=zeros(matrix_dim,1);
    PJM=JM;
    PV=VM;
    for i=1:1:matrix_dim
        JR(i)=i;
        for j=JM(i):JM(i+1)-1
            JC = JM(j);
            if JC > i && JD(i) == 0
                JD(i) = j ;
            end
        end
        if JD(i) == 0
            JD(i) = j ;
        end
        for j = JM(i):(JD(i)-1)
            jc = JM(j);
            PV(j) = PV(j) / PV(jc);
            for jj = JD(jc):(JM(jc+1)-1)
                jk = JR(JM(jj));
                if jk ~= 0
                    PV(jk) = PV(jk) - PV(j) * PV(jj);
                end
            end
        end
        JR(i) = 0;
        for j=JM(i):JM(i+1)-1 
            JR(JM(j)) = 0;
        end 
    end
end

% Applied ILU(0) Preconditioning
% by Using Forward and Backward Substitions for MSR
function x = ilu0_forward_backward_substitution(PV, PJM, answer, matrix_dim,x)
    size_x=size(x,1); 
    x=zeros(size_x,1);
    % Forward Substitution (Ly = b)
    y = zeros(size_x, 1);
    y(1) = answer(1);
    for i = 2:1:size_x
        if (PJM(i)==length(PV) || PJM(i)<=length(PV)) && PJM(i)~=PJM(i+1)
            for j = PJM(i):(PJM(i + 1) - 1)
                jc = PJM(j); % Column index: PV(j) = a(i, jc)
                y(i) = y(i) - PV(j) * y(jc);
            end
            y(i) = y(i) + answer(i);
        end
    end

    % Backward Substitution (Ux = y)
    x(size_x) = y(size_x) / PV(PJM(size_x + 1) - 1);

    for i = size_x-1:-1:1
        for j = PJM(i):(PJM(i + 1) - 2)
            jc = PJM(j); % Column index: PV(j) = a(i, jc)
            x(i) = x(i) - PV(j) * x(jc);
        end
        x(i) = (y(i) + x(i)) / PV(PJM(i + 1) - 1);
    end
end

function GMRES_figures(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, r0)
    [x, res, ort, rho] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 0, 0, r0, 0);
    
    figure
    semilogy(abs(res))
    ylim([1e-8 1])
    xlim([1 530])
    hold on
    xlabel('Number of Iterations');
    ylabel('Residual')
    title('Norms vs. Iteration Number')

    [x1, res1, ort1, rho1] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 0, r0, 0);
    disp(size(res1))
    semilogy(abs(res1), 'r')
    hold on
    [x2, res2, ort2, rho2] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 1, r0, 0);
    semilogy(abs(res2))
    [x3, res3, ort3, rho3] = GMRES(JM, VM, V, H, matrix_dim, symmetry, x0, g, m, tol, 1, 2, r0, 0);
    semilogy(abs(res3))
    legend('No Precondition', 'Jacobi Preconditioning', 'Gauss-Seidel Preconditioning', 'ILU(0) Preconditioning')
    ylim([1e-8 1])
    title('Relative Residual vs. Iteration Number: Analysis of Convergence')
    figure
    semilogy(abs(ort), 'k')
    xlim([0 530])
    xlabel('Number of Iterations');
    ylabel('(v_1, v_k)')
    title("Orthogonality of Krylov Vectors vs. Iteration Number: Convergence Analysis")
end

