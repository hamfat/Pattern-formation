
%model parameters
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi a b
d=0.0001; %diffusion coefficient
glbar=0.25;
gkbar=1.0;
gcabar=0.4997;
vlbar=-0.875;
vkbar=-1.125;
va=3.1999;
vb=-0.75;  %as bifurcation parameter vb=[-1.184,-0.464]
vc=5.5172;
vd=-0.7586;
psi=0.1665;

A_0=.2;
% solve for v_0 N_0
% VN= fsolve(@(X) dimless(X),[-0.6848;0.0024]);
% V_0 = VN(1);
% N_0 = VN(2);

tf=200;
Tf =1000;
tspan=linspace(0,tf,Tf);
%V_0=-0.7105;%0.7164;
%N_0=0.0018;%1679;

M =1000;
sigma=0.5;
a=-1.0;
b=1.0 ;
x=linspace(a,b,M);
% v_initial=V_0*ones(size(x))+A_0*(tanh(x+2*sigma)-tanh(x-2*sigma)).*ones(size(x));
%v_initial=V_0*ones(size(x))+ A_0*sech(pi*x).*cos(pi*x).*ones(size(x));
v_initial=-0.7568*ones(size(x))+A_0*exp(-((x-.0)/sigma).^2).*ones(size(x));
%A_0*cos(2*pi*x/4).*ones(size(x)); A_0*sech(pi*x),
% A_0*exp(-10*(sin(x-0/2)).^2),,A_0*sech(pi*x).*cos(pi*x)

n_initial=0.002*ones(size(x));

initial=[v_initial, n_initial];

%F1 = @(t,x) myworkinternal(t,x,M);
%F1(0,initial)

[tsol,xsol]=ode15s( @(t,x) myworkinternal(t,x,M),tspan,initial);

V=xsol(:,1:M)';
N=xsol(:,M+1:end)';
[T,X] = meshgrid(tsol,x);
dx = (b - a)/(M - 1);
VX=(V(:,2:M)-V(:,1:M-1))/dx;




c=0.03;




figure(1)
subplot(1,2,1);
mesh(T,X,V,'FaceLighting','gouraud','LineWidth',0.5)
xlabel('\itT\rm')
ylabel('\itX\rm')
zlabel('\itV\rm')
set ( gca, 'xdir', 'reverse' )
%title('v_{b}=-0.8999')
%grid on
 %colorbar
subplot(1,2,2);
mesh(T,X,N,'FaceLighting','gouraud','LineWidth',0.5)
xlabel('\itT\rm')
ylabel('\itX\rm')
zlabel('\itN\rm')
set ( gca, 'xdir', 'reverse' )
%   figure(2)
%   surf(T,X,V,'EdgeColor', 'none')


 figure(15)
 plot3(V(:,500),VX(:,500),N(:,500))
  xlabel('\itV\rm')
 ylabel('\itV_x\rm')
  zlabel('\itN\rm')
% figure(2)
% mesh(T(:,1:M-1),X(:,1:M-1),VX,'FaceLighting','gouraud','LineWidth',0.5)
% xlabel('\itT\rm')
% ylabel('\itX\rm')
% zlabel('\itV_x\rm')
% set ( gca, 'xdir', 'reverse' )
% 
% figure(3)
% subplot(1,3,1)
%  plot(x,V(:,500))
%  ylabel('\itV\rm')
%   xlabel('\itX\rm')
%  subplot(1,3,2)
% plot(x,N(:,500))
% ylabel('\itN\rm')
%  xlabel('\itX\rm')
%  subplot(1,3,3)
%  plot(x,VX(:,500))
%  ylabel('\itV_x\rm')
%  xlabel('\itX\rm')
%  
%  figure(4)
% mesh(X-c*T,T,V,'FaceLighting','gouraud','LineWidth',0.5)
% xlabel('\itT\rm')
% ylabel('\itX-cT\rm')
% zlabel('\itV\rm')
% set ( gca, 'xdir', 'reverse' )
% %title('v_{b}=-0.8999')
% %grid on
%  %colorbar
%  
% figure(5)
% mesh(X-c*T,T,N,'FaceLighting','gouraud','LineWidth',0.5)
% xlabel('\itT\rm')
% ylabel('\itX-cT\rm')
% zlabel('\itN\rm')
% set ( gca, 'xdir', 'reverse' )
% %title('v_{b}=-0.8999')
% %grid on
%  %colorbar
%  
% figure(6)
% mesh(X(:,1:M-1)-c*T(:,1:M-1),T(:,1:M-1),VX,'FaceLighting','gouraud','LineWidth',0.5)
% xlabel('\itT\rm')
% ylabel('\itX-cT\rm')
% zlabel('\itV_x\rm')
% set ( gca, 'xdir', 'reverse' )
 


function xdot = myworkinternal(t,x,M)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi a b

V = x(1:M);
N = x(M+1:end);

 a = -1.0;
 b = 1.0;
dx=(b-a)/(M-1);
mu=d/dx^2;

dVdt = zeros(M,1);
dNdt = zeros(M,1);

for i=1:M
    minf=0.5*(1+tanh(V(i)*va-vb));
    ninf=0.5*(1+tanh(V(i)*vc-vd));
    lambda_v=cosh((V(i)*vc-vd)/2);
    
    if i==1
			dVdt(i)= 2*mu*(V(i+1)-V(i))-(glbar*(V(i)-vlbar)+gkbar*N(i)*(V(i)-vkbar)+gcabar*minf*(V(i)-1));

			dNdt(i)= psi*lambda_v*(ninf-N(i));
    end
    if i==M
			dVdt(i)= 2*mu*(V(i-1)-V(i))-(glbar*(V(i)-vlbar)+gkbar*N(i)*(V(i)-vkbar)+gcabar*minf*(V(i)-1));

			dNdt(i)= psi*lambda_v*(ninf-N(i));
    end
    if i>1 && i<M
			dVdt(i)= mu*(V(i+1)-2*V(i)+V(i-1))-(glbar*(V(i)-vlbar)+gkbar*N(i)*(V(i)-vkbar)+gcabar*minf*(V(i)-1));
			dNdt(i)= psi*lambda_v*(ninf-N(i));
    end
end

xdot = [dVdt;dNdt];





end

function ica = I_Ca(V)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
    ica=gcabar * m_inf(V)*(V - 1);
end

function ik = I_K(V,N)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
    ik=gkbar  * N * (V - vkbar);
end
function ileak = I_Leak(V)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
    ileak=glbar* (V - vlbar);
end

function VN = dimless(X)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
    V=X(1);
    N=X(2);
    
    VN(1) = - I_Leak(V) - I_K(V,N) - I_Ca(V);
    VN(2) = psi*lamda(V)*(n_inf(V) - N) ;
end


function out = m_inf(V)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
	out = 0.5*(1+tanh((V*va-vb)));

end

function out =Ca(V)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
	out = -eta*m_inf(V)*(V*vca-vca);
end

function out =n_inf(V)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
	out = 0.5*(1+tanh(V*vc-vd));
end

function out =lamda(V)
global d glbar gkbar gcabar vlbar vkbar va vb vc vd psi
	out = cosh((V*vc-vd)/2);
end


