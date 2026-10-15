// Gmsh project 

//************************************************************//
// GEOMETRY PARAMETERS
//************************************************************//

// CASE
aoa=10.*Pi/180; // working AoA (guides wake direction)

// AIRFOIL
c=1; // chord
t=0.12; // max thickness (in % of chord)
m=0.00; // max camber (in % of chord)
p=0.04; // max camber position (in % of chord)
tetype=0; // trailing edge type: 0: sharp; 1: blunt (not implemented);
N=101; // number of points defining airfoil

// NEARWALL (defined as orthogonal distance from airfoil surface)
//Rnf=2*c; // radius
//tnwle=0.2*c; // near wall region thickness at leading edge (in chord lengths)
//eu=1.0; au=1.0; // tu=tnwle*(1+au*(x/c)^eu) [au<0 for tangency with wake]
//el=1.0; al=1.0; // tl=tnwle*(1+al*(x/c)^el) [au<0 for tangncy with wake]
Rnf=0.5*c; anf=2*aoa/3;

// MIDFIELD (defined as orthogonal distance from airfoil surface)
Rmf=1.0*c; amf=aoa/3;

// WAKE
daoau=(2*aoa/3);//*Pi/180; // upper wake opening angle (angle_upper=aoa+daoau)
daoauu=(3*aoa/4);//*Pi/180; // upper upper wake opening angle (angle_upper=aoa+daoau)
daoal=(aoa/2);//*Pi/180; // lower wake opening angle (angle_lower=aoa-daoal)
daoall=(2*aoa/3);//*Pi/180; // lower lower wake opening angle (angle_lower=aoa-daoal)

// BOUNDARY
a=7*c;//10*c; // upstream dimensions from trailing edge (in chord lengths)
b=10*c;//10*c; // downstream dimensions from trailing edge (in chord lengths)
NP=101; // number of points defining semi-circle
fr=0.1*c; // fringe region length (<0 for no fringe region)
ajint=3*Pi/4;

// JET (Currently useless)
pj=.1; // jet location (in % of chord)
tj=.01; // jet thickness (in % of chord)
ajf=5*Pi/180; // jet semi-opening angle forward
ajb=5*Pi/180; // jet semi-opening angle backward

//************************************************************//
// MESH PARAMETERS
//************************************************************//

// GENERAL
nord=2; // Polynomial order for curved edges
ls=0.01; // Characteristic length (useless for structured mesh)

Ph=3; // higher order
Pl=3; // lower order
ikeepl1k=0; // 0: keep 1st distance; 1: keep progression ratio

// AIRFOIL
Nua=35; // number of points along airfoil upper surface
ibuas=2;//2; // 0: equispaced; 1: Progression; 2: Bump
kbuas=0.4; // Progression/Bump parameter
Nuf=20; // number of points along airfoil upper surface
ibufs=1;//1; // 0: equispaced; 1: Progression; 2: Bump
kbufs=1.1; // Progression/Bump parameter
Nla=35; // number of points along airfoil lower surface
iblas=2;//2; // 0: equispaced; 1: Progression; 2: Bump
kblas=0.4; // Progression/Bump parameter
Nlf=20; // number of points along airfoil lower surface
iblfs=1;//1; // 0: equispaced; 1: Progression; 2: Bump
kblfs=1.1; // Progression/Bump parameter

// NEARFIELD REGION
ibuanf=ibuas;//1; // 0: equispaced; 1: Progression; 2: Bump
kbuanf=kbuas;//1.02; // Progression/Bump parameter
ibufnf=0; // 0: equispaced; 1: Progression; 2: Bump
kbufnf=1.00; // Progression/Bump parameter
iblanf=0;//1; // 0: equispaced; 1: Progression; 2: Bump
kblanf=0.00; // Progression/Bump parameter
iblfnf=0; // 0: equispaced; 1: Progression; 2: Bump
kblfnf=1.00; // Progression/Bump parameter

Puanf=Ph; // target polynomial order for upper nearfield +1 (=NUMMODES)
kutenf=1.25; // wall-normal target progression ratio (virtual)
xfirstu=4e-6; // upper trailing edge first point distance
Pufnf=Ph;

Planf=Ph; // target polynomial order for lower nearfield +1 (=NUMMODES)
xfirstl=1*xfirstu;//0.5*xfirstu; // lower trailing edge first point distance

xfirstle=xfirstu*0.5; // leading edge first point distance
xfirstuj=(pj)*xfirstu+(1-pj)*xfirstle; // leading edge first point distance
xfirstlj=(pj)*xfirstl+(1-pj)*xfirstle; // leading edge first point distance

// MIDFIELD REGION
ibuamf=ibuanf;//1; // 0: equispaced; 1: Progression; 2: Bump
kbuamf=kbuanf; // Progression/Bump parameter
ibufmf=0; // 0: equispaced; 1: Progression; 2: Bump
kbufmf=1.00; // Progression/Bump parameter
iblamf=0;//1; // 0: equispaced; 1: Progression; 2: Bump
kblamf=1.00; // Progression/Bump parameter
iblfmf=0; // 0: equispaced; 1: Progression; 2: Bump
kblfmf=1.00; // Progression/Bump parameter

Puamf=Pl; // target polynomial order for upper midfield +1 (=NUMMODES)
kutemf=1.3; // wall-normal target progression ratio (virtual)
Pufmf=Pl;

Plamf=Pl; // target polynomial order for lower midfield +1 (=NUMMODES)
Plfmf=Pl;

// FARFIELD REGION
Pff=Pl;
kuteff=1.3; // progression ratio in upper farfield (issued from trailing edge)

// WAKE REGION
Pw=Ph;
kw=1.01;//1.005 // target progression ratio along wake

kudnf=1.05; // target vertical progression ratio upper wake aft (kutenf^0.5?)
kldnf=1.05; // target vertical progression ratio lower wake aft (kfl^0.5?)

Puwmf=Ph;
kudmf=1.05; // target vertical progression ratio upper midfield wake
Plwmf=Pl;
kldmf=1.05; // target vertical progression ratio lower midfield wake

// FRINGE REGION
Pii=Ph;
ki=1.00;

//// BOUNDARY
//kfPUf=1;//1.001; // Progression ratio along upper boundary forward
//kfPLf=1;//1.001; // Progression ratio along lower boundary forward


////////////////////////////////////////////////////////////
///// NEWTON ITERATION TO COMPUTE K
////////////////////////////////////////////////////////////
Macro Compk
kg3=1+1.5/(n-3)*(-1+Sqrt(1+(8/3)*(lt-l1*(n-1))*(n-3)/((n-1)*(n-2))));
Printf('kguess=%g',kg3);
If (kguess<0)
  If (kg3>1)
    kguess=1.8; // handy guess
    //kguess=1+2*(lt-l1*(n-1))/((n-1)*(n-2)); // Taylor second order
    //kguess=1+1.5/(n-3)*(-1+Sqrt(1+(8/3)*(lt-l1*(n-1))*(n-3)/((n-1)*(n-2)))); // Taylor third order
    //kguess=kguess*1.2;
    Else
    kguess=1/1.1;
  EndIf
EndIf
dk=1;
For ig In {1:50}
  fguess=l1*kguess^(n-1)-lt*kguess+(lt-l1);
  dfdk=l1*(n-1)*kguess^(n-2)-lt;
  dk=-fguess/dfdk;
//  Printf('%g %g %g %g',ig,kguess,fguess,dk);
  kguess=kguess+dk;
//  If (Fabs(dk)<1e-5)
//    EndFor
//  EndIf
EndFor
fguess=l1*kguess^(n-1)-lt*kguess+(lt-l1);
Printf('i=%g; k=%g; f=%g',ig,kguess,fguess);
Return
////////////////////////////////////////////////////////////

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* * * SET GEOMETRY
//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
Geometry.Tolerance=1e-14;
Printf('Geometrical Tolerance: %g',Geometry.Tolerance);
View.Visible=1;
Printf('%g',View.Visible);
Geometry.Normals=1;
Geometry.Points=1;
Geometry.PointNumbers=0;
Geometry.Color.Points={90,90,90};
Geometry.Lines=1;
Geometry.LineNumbers=0;
Geometry.LineWidth=2;
//Geometry.Surfaces=1;
//Geometry.SurfaceNumbers=1;
//Geometry.AutoCoherence=2; // remove degenerate entities
Mesh.CharacteristicLengthMin=0;
Mesh.CharacteristicLengthMax=1e22;
Mesh.Points=0;
Mesh.PointNumbers=0;
Mesh.Lines=1;
Mesh.LineNumbers=0;
Mesh.Smoothing=1;
Mesh.SurfaceEdges=1;
Mesh.SurfaceFaces=0;
Mesh.SurfaceNumbers=0;

Mesh.Quadrangles=1; // display
Mesh.Triangles=1;

Mesh.SecondOrderIncomplete=1; // Generate 6(8)-node triangles (quads) 
Mesh.SecondOrderExperimental=0;
Mesh.SecondOrderLinear=0; // 1: Linear interpolation (ignore surfaces)


// NACA 4-DIGIT AIRFOIL COEFFICIENTS
a0=0.2969; a1=-0.1260; a2=-0.3516; a3=0.2843; // 4-digit NACA coeff
If (tetype==1)
  a4=-0.1015; // blunt trailing edge
  Else
  a4=-0.1036; // sharp trailing edge
EndIf

////* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//// NEARWALL
//If (au<0)
//  slope_tte=5*t*(a0/2+a1+2*a2+3*a3+4*a4);
//  slope_cte=-2*m/(1-p);
//  slope_te=Tan(daoau+aoa-(Atan(slope_tte)+Atan(slope_cte)));
//  au=c*slope_te/(eu*tnwle); // set amplification factor
//EndIf
//If (al<0)
//  slope_tte=-5*t*(a0/2+a1+2*a2+3*a3+4*a4);
//  slope_cte=-2*m/(1-p);
//  slope_te=Tan(-daoal+aoa-(Atan(slope_tte)+Atan(slope_cte)));
//  al=-c*slope_te/(el*tnwle); // set amplification factor
//EndIf
//tnwlte=tnwle*(1+al);
//tnwute=tnwle*(1+au);

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* * * COMPUTE GEOMETRY
//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* AIRFOIL, NEARWALL AND JET SLOT
//* jet location
xja = c*(pj+tj/2);
ytja = 5*t*c*(a0*Sqrt(xja/c)+a1*(xja/c)+a2*(xja/c)^2+a3*(xja/c)^3+a4*(xja/c)^4);
dytdxja = 5*t*(a0/(2*Sqrt(xja/c))+a1+2*a2*(xja/c)+3*a3*(xja/c)^2+4*a4*(xja/c)^3);
ycja = m*(xja/p^2)*(2*p-xja/c); // jet assumed on front camber law
dycdxja = (2*m/p^2)*(p-xja/c);
thja = Atan(dycdxja); xuja = xja-ytja*Sin(thja); yuja = ycja+ytja*Cos(thja);
xjf = c*(pj-tj/2);
ytjf = 5*t*c*(a0*Sqrt(xjf/c)+a1*(xjf/c)+a2*(xjf/c)^2+a3*(xjf/c)^3+a4*(xjf/c)^4);
dytdxjf = 5*t*(a0/(2*Sqrt(xjf/c))+a1+2*a2*(xjf/c)+3*a3*(xjf/c)^2+4*a4*(xjf/c)^3);
ycjf = m*(xjf/p^2)*(2*p-xjf/c); // jet assumed on front camber law
dycdxjf = (2*m/p^2)*(p-xjf/c);
thjf = Atan(dycdxjf); xujf = xjf-ytjf*Sin(thjf); yujf = ycjf+ytjf*Cos(thjf);
//Printf("%g %g", ycja, ycjf);

//* singular points

xte0 = 0; yte0 = 0;
xte = xte0; yte = yte0;
teP=newp; Point(teP)={xte,yte,0};

xle0 = -c; yle0 = 0;
xle = -c*Cos(aoa); yle = c*Sin(aoa);
leP=newp; Point(leP)={xle,yle,0};

xj = c*pj;
ytj = 5*t*c*(a0*Sqrt(xj/c)+a1*(xj/c)+a2*(xj/c)^2+a3*(xj/c)^3+a4*(xj/c)^4);
dytdxj = 5*t*(a0/(2*Sqrt(xj/c))+a1+2*a2*(xj/c)+3*a3*(xj/c)^2+4*a4*(xj/c)^3);
ycj = m*(xj/p^2)*(2*p-xj/c); // jet assumed on front camber law
dycdxj = (2*m/p^2)*(p-xj/c);
thj = Atan(dycdxj);

xuj0 = -c+xj-ytj*Sin(thj); yuj0 = ycj+ytj*Cos(thj);
xuj = xte0+(xuj0-xte0)*Cos(aoa)+(yuj0-yte0)*Sin(aoa);
yuj = yte0-(xuj0-xte0)*Sin(aoa)+(yuj0-yte0)*Cos(aoa);
ujP=newp; Point(ujP)={xuj,yuj,0};

xlj0 = -c+xj+ytj*Sin(thj); ylj0 = ycj-ytj*Cos(thj);
xlj = xte0+(xlj0-xte0)*Cos(aoa)+(ylj0-yte0)*Sin(aoa);
ylj = yte0-(xlj0-xte0)*Sin(aoa)+(ylj0-yte0)*Cos(aoa);
ljP=newp; Point(ljP)={xlj,ylj,0};


dycdxte = 2*m/(p-1); dytdxte = 5*t*(a0/2+a1+2*a2+3*a3+4*a4);
dycdxle=2*m/p; thnfle=Atan(dycdxle)+Pi/2;

//* loop over chord
xp=1.1*c; iL=-1; iU=-1;
For i In {0 : N-1}

  //* chord line definition
  xc = c*(1+Cos((i*Pi)/N))/2;
  //Printf('%g %g %g %g',xc,pj*c,xp,pj*c);
  If ((xc < pj*c) && (xp >= pj*c))
    xj = pj*c; // relocate point on jet position
    ij = i; // save jet position index
  EndIf
  x[i] = xc; // store point
  xp = x[i];

    //* thickness (symmetric airfoil)
  yt = 5*t*c*(a0*Sqrt(x[i]/c)+a1*(x[i]/c)+a2*(x[i]/c)^2+a3*(x[i]/c)^3+a4*(x[i]/c)^4);
  If (x[i] != 0)
    dytdx[i] = 5*t*(a0/(2*Sqrt(x[i]/c))+a1+2*a2*(x[i]/c)+3*a3*(x[i]/c)^2+4*a4*(x[i]/c)^3);
  EndIf

  //* camber
  If (x[i] <= p) // camber for x/c <= p
    yc = m*(x[i]/p^2)*(2*p-x[i]/c);
    dycdx[i] = (2*m/p^2)*(p-x[i]/c);
    Else // camber for x/c > p
    yc = m*((c-x[i])/(1-p)^2)*(1+x[i]/c-2*p);
    dycdx[i] = (2*m/(1-p)^2)*(p-x[i]/c);
  EndIf

  //* airfoil (x,y) coordinates (origin at trailing edege)
  th = Atan(dycdx[i]);
  xu0 = -c+x[i]-yt*Sin(th);
  yu0 = yc+yt*Cos(th);
  xl0 = -c+x[i]+yt*Sin(th);
  yl0 = yc-yt*Cos(th);
  xu[i] = xte+(xu0-xte)*Cos(aoa)+(yu0-yte)*Sin(aoa);
  yu[i] = yte-(xu0-xte)*Sin(aoa)+(yu0-yte)*Cos(aoa);
  xl[i] = xte+(xl0-xte)*Cos(aoa)+(yl0-yte)*Sin(aoa);
  yl[i] = yte-(xl0-xte)*Sin(aoa)+(yl0-yte)*Cos(aoa);

  //* generate points
  iL = iL+1; lsP[iL] = newp; Point(lsP[iL]) = {xl[i],yl[i],0};
  iU=iU+1; usP[iU] = newp; Point(usP[iU]) = {xu[i], yu[i], 0, ls};
EndFor

uasL = newl; Spline(uasL) = {teP,usP[{1:ij-1}],ujP};
ufsL = newl; Spline(ufsL) = {ujP,usP[{ij:#usP[]-1}],leP};
lasL = newl; Spline(lasL) = {teP,lsP[{1:ij-1}],ljP};
lfsL = newl; Spline(lfsL) = {ljP,lsP[{ij:#lsP[]-1}],leP};

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* NEARFIELD REGION

// semicircle definition
xcnf0=-(1-pj)*c; ycnf0=0;
xcnf=xcnf0*Cos(anf)+ycnf0*Sin(anf); ycnf=-xcnf0*Sin(anf)+ycnf0*Cos(anf);
cnfP=newp; Point(cnfP)={xcnf,ycnf,0};
xunf0=xcnf0; yunf0=ycnf0+Rnf;
xunf=xunf0*Cos(anf)+yunf0*Sin(anf); yunf=-xunf0*Sin(anf)+yunf0*Cos(anf);
unfP=newp; Point(unfP)={xunf,yunf,0};
xlnf0=xcnf0; ylnf0=ycnf0-Rnf;
xlnf=xlnf0*Cos(anf)+ylnf0*Sin(anf); ylnf=-xlnf0*Sin(anf)+ylnf0*Cos(anf);
lnfP=newp; Point(lnfP)={xlnf,ylnf,0};

// leading edge intersect
aa=Tan(Atan(dycdxle)+Pi/2+Pi/2);
bb=yle0-aa*xle0;
aaux=1+aa^2; baux=2*(aa*(bb-ycnf0)-xcnf0); caux=(bb-ycnf0)^2+xcnf0^2-Rnf^2;
xlenf0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
ylenf0=aa*xlenf0+bb;
xlenf=xlenf0*Cos(anf)+ylenf0*Sin(anf); ylenf=-xlenf0*Sin(anf)+ylenf0*Cos(anf);
lenfP=newp; Point(lenfP)={xlenf,ylenf,0};

lenfL=newl; Line(lenfL)={leP,lenfP};

// upper jet intersect
//aa=Tan((1-Rnf/a)*(Atan(dycdxj)+Atan(dytdxj))+(Rnf/a)*(ajint-Pi/2)+Pi/2);
aa=Tan((2/3)*(Atan(dycdxj)+Atan(dytdxj))+(1/3)*(ajint-Pi/2)+Pi/2);
//aa=Tan(Atan(dycdxj)+Atan(dytdxj)+Pi/2);
bb=yuj0-aa*xuj0;
aaux=1+aa^2; baux=2*(aa*(bb-ycnf0)-xcnf0); caux=(bb-ycnf0)^2+xcnf0^2-Rnf^2;
xujnf0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
Printf('%g %g',xujnf0,xunf0);
If (xujnf0 < xunf0)
  yujnf0=aa*xujnf0+bb;
  xujnf=xujnf0*Cos(anf)+yujnf0*Sin(anf); yujnf=-xujnf0*Sin(anf)+yujnf0*Cos(anf);
  ujnfP=newp; Point(ujnfP)={xujnf,yujnf,0};
  ufnfL=newl; Circle(ufnfL) = {lenfP,cnfP,ujnfP};
  iujf=1;
  Else
  yujnf0=yunf0;
  xujnf0=(yujnf0-bb)/aa;
  xujnf=xujnf0*Cos(anf)+yujnf0*Sin(anf); yujnf=-xujnf0*Sin(anf)+yujnf0*Cos(anf);
  ujnfP=newp; Point(ujnfP)={xujnf,yujnf,0};
  ufnfL1=newl; Circle(ufnfL1) = {lenfP,cnfP,unfP};
  ufnfL2=newl; Line(ufnfL2) = {unfP,ujnfP};
  ufnfL=newl; Compound Line(ufnfL) = {ufnfL1,ufnfL2};
  iujf=0;
EndIf
ujnfL=newl; Line(ujnfL)={ujP,ujnfP};

// lower jet intersect
//aa=Tan((1-Rnf/a)*(Atan(dycdxj)-Atan(dytdxj))+(Rnf/a)*(-ajint+Pi/2)+Pi/2);
aa=Tan((2/3)*(Atan(dycdxj)-Atan(dytdxj))+(1/3)*(-ajint+Pi/2)+Pi/2);
//aa=Tan(Atan(dycdxj)-Atan(dytdxj)+Pi/2);
bb=ylj0-aa*xlj0;
aaux=1+aa^2; baux=2*(aa*(bb-ycnf0)-xcnf0); caux=(bb-ycnf0)^2+xcnf0^2-Rnf^2;
xljnf0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
Printf('%g %g',xljnf0,xlnf0);
If (xljnf0 < xlnf0)
  yljnf0=aa*xljnf0+bb;
  xljnf=xljnf0*Cos(anf)+yljnf0*Sin(anf); yljnf=-xljnf0*Sin(anf)+yljnf0*Cos(anf);
  ljnfP=newp; Point(ljnfP)={xljnf,yljnf,0};
  lfnfL=newl; Circle(lfnfL) = {lenfP,cnfP,ljnfP};
  iljf=1;
  Else
  yljnf0=ylnf0;
  xljnf0=(yljnf0-bb)/aa;
  xljnf=xljnf0*Cos(anf)+yljnf0*Sin(anf); yljnf=-xljnf0*Sin(anf)+yljnf0*Cos(anf);
  ljnfP=newp; Point(ljnfP)={xljnf,yljnf,0};
  lfnfL1=newl; Circle(lfnfL1) = {lenfP,cnfP,lnfP};
  lfnfL2=newl; Line(lfnfL2) = {lnfP,ljnfP};
  lfnfL=newl; Compound Line(lfnfL) = {lfnfL1,lfnfL2};
  iljf=0;
EndIf
ljnfL=newl; Line(ljnfL)={ljP,ljnfP};

// upper trailing edge intersect
aa=Tan(Atan(dycdxte)+Atan(dytdxte)+Pi/2);
bb=yte0-aa*xte0;
aaux=1+aa^2; baux=2*(aa*(bb-ycnf0)-xcnf0); caux=(bb-ycnf0)^2+xcnf0^2-Rnf^2;
yutenf0=yunf0;
xutenf0=(yutenf0-bb)/aa;
xutenf=xutenf0*Cos(anf)+yutenf0*Sin(anf); yutenf=-xutenf0*Sin(anf)+yutenf0*Cos(anf);
utenfP=newp; Point(utenfP)={xutenf,yutenf,0};
If (iujf == 0)
  uanfL=newl; Line(uanfL) = {ujnfP,utenfP};
  Else 
  //uanfL1=newl; Circle(uanfL1) = {ujnfP,cnfP,unfP};
  //uanfL2=newl; Line(uanfL2) = {unfP,utenfP};
  uanfL=newl; Line(uanfL) = {ujnfP,unfP,utenfP};
  //Compound Line(uanfL) = {uanfL1,uanfL2};
  //Compound Curve(uanfL) = {uanfL1,uanfL2};
EndIf
utenfL=newl; Line(utenfL)={teP,utenfP};

// lower trailing edge intersect
aa=Tan(Atan(dycdxte)-Atan(dytdxte)+Pi/2);
bb=yte0-aa*xte0;
aaux=1+aa^2; baux=2*(aa*(bb-ycnf0)-xcnf0); caux=(bb-ycnf0)^2+xcnf0^2-Rnf^2;
yltenf0=ylnf0;
xltenf0=(yltenf0-bb)/aa;
xltenf=xltenf0*Cos(anf)+yltenf0*Sin(anf); yltenf=-xltenf0*Sin(anf)+yltenf0*Cos(anf);
ltenfP=newp; Point(ltenfP)={xltenf,yltenf,0};
If (iljf == 0)
  lanfL=newl; Line(lanfL) = {ljnfP,ltenfP};
  Else 
  //lanfL1=newl; Circle(lanfL1) = {ljnfP,cnfP,lnfP};
  //lanfL2=newl; Line(lanfL2) = {lnfP,ltenfP};
  lanfL=newl; Line(lanfL) = {ljnfP,lnfP,ltenfP};
  //Compound Line(lanfL) = {lanfL1,lanfL2};
  //Compound Curve{lanfL1,lanfL2};
EndIf
ltenfL=newl; Line(ltenfL)={teP,ltenfP};


//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* MIDFIELD REGION

// semicircle definition
xcmf0=-(1-pj)*c; ycmf0=0;
xcmf=xcmf0*Cos(amf)+ycmf0*Sin(amf); ycmf=-xcmf0*Sin(amf)+ycmf0*Cos(amf);
cmfP=newp; Point(cmfP)={xcmf,ycmf,0};
xumf0=xcmf0; yumf0=ycmf0+Rmf;
xumf=xumf0*Cos(amf)+yumf0*Sin(amf); yumf=-xumf0*Sin(amf)+yumf0*Cos(amf);
umfP=newp; Point(umfP)={xumf,yumf,0};
xlmf0=xcmf0; ylmf0=ycmf0-Rmf;
xlmf=xlmf0*Cos(amf)+ylmf0*Sin(amf); ylmf=-xlmf0*Sin(amf)+ylmf0*Cos(amf);
lmfP=newp; Point(lmfP)={xlmf,ylmf,0};

// leading edge intersect
aa=Tan(Atan(dycdxle)+Pi/2+Pi/2);
bb=yle0-aa*xle0;
aaux=1+aa^2; baux=2*(aa*(bb-ycmf0)-xcmf0); caux=(bb-ycmf0)^2+xcmf0^2-Rmf^2;
xlemf0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
ylemf0=aa*xlemf0+bb;
xlemf=xlemf0*Cos(amf)+ylemf0*Sin(amf); ylemf=-xlemf0*Sin(amf)+ylemf0*Cos(amf);
lemfP=newp; Point(lemfP)={xlemf,ylemf,0};

lemfL=newl; Line(lemfL)={lenfP,lemfP};

// upper jet intersect
//aa=Tan((1-Rmf/a)*(Atan(dycdxj)+Atan(dytdxj))+(Rmf/a)*(ajint-Pi/2)+Pi/2);
aa=Tan((1/3)*(Atan(dycdxj)+Atan(dytdxj))+(2/3)*(ajint-Pi/2)+Pi/2);
//aa=Tan(Atan(dycdxj)+Atan(dytdxj)+Pi/2);
bb=yuj0-aa*xuj0;
aaux=1+aa^2; baux=2*(aa*(bb-ycmf0)-xcmf0); caux=(bb-ycmf0)^2+xcmf0^2-Rmf^2;
xujmf0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
Printf('%g %g',xujmf0,xumf0);
If (xujmf0 < xumf0)
  yujmf0=aa*xujmf0+bb;
  xujmf=xujmf0*Cos(amf)+yujmf0*Sin(amf); yujmf=-xujmf0*Sin(amf)+yujmf0*Cos(amf);
  ujmfP=newp; Point(ujmfP)={xujmf,yujmf,0};
  ufmfL=newl; Circle(ufmfL) = {lemfP,cmfP,ujmfP};
  iujf=1;
  Else
  yujmf0=yumf0;
  xujmf0=(yujmf0-bb)/aa;
  xujmf=xujmf0*Cos(amf)+yujmf0*Sin(amf); yujmf=-xujmf0*Sin(amf)+yujmf0*Cos(amf);
  ujmfP=newp; Point(ujmfP)={xujmf,yujmf,0};
  ufmfL1=newl; Circle(ufmfL1) = {lemfP,cmfP,umfP};
  ufmfL2=newl; Line(ufmfL2) = {umfP,ujmfP};
  ufmfL=newl; Compound Line(ufmfL) = {ufmfL1,ufmfL2};
  iujf=0;
EndIf
ujmfL=newl; Line(ujmfL)={ujnfP,ujmfP};

// lower jet intersect
//aa=Tan((1-Rmf/a)*(Atan(dycdxj)-Atan(dytdxj))+(Rmf/a)*(-ajint+Pi/2)+Pi/2);
aa=Tan((1/3)*(Atan(dycdxj)-Atan(dytdxj))+(2/3)*(-ajint+Pi/2)+Pi/2);
//aa=Tan(Atan(dycdxj)-Atan(dytdxj)+Pi/2);
bb=ylj0-aa*xlj0;
aaux=1+aa^2; baux=2*(aa*(bb-ycmf0)-xcmf0); caux=(bb-ycmf0)^2+xcmf0^2-Rmf^2;
xljmf0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
Printf('%g %g',xljmf0,xlmf0);
If (xljmf0 < xlmf0)
  yljmf0=aa*xljmf0+bb;
  xljmf=xljmf0*Cos(amf)+yljmf0*Sin(amf); yljmf=-xljmf0*Sin(amf)+yljmf0*Cos(amf);
  ljmfP=newp; Point(ljmfP)={xljmf,yljmf,0};
  lfmfL=newl; Circle(lfmfL) = {lemfP,cmfP,ljmfP};
  iljf=1;
  Else
  yljmf0=ylmf0;
  xljmf0=(yljmf0-bb)/aa;
  xljmf=xljmf0*Cos(amf)+yljmf0*Sin(amf); yljmf=-xljmf0*Sin(amf)+yljmf0*Cos(amf);
  ljmfP=newp; Point(ljmfP)={xljmf,yljmf,0};
  lfmfL1=newl; Circle(lfmfL1) = {lemfP,cmfP,lmfP};
  lfmfL2=newl; Line(lfmfL2) = {lmfP,ljmfP};
  lfmfL=newl; Compound Line(lfmfL) = {lfmfL1,lfmfL2};
  iljf=0;
EndIf
ljmfL=newl; Line(ljmfL)={ljnfP,ljmfP};

// upper trailing edge intersect
aa=Tan(Atan(dycdxte)+Atan(dytdxte)+Pi/2);
bb=yte0-aa*xte0;
aaux=1+aa^2; baux=2*(aa*(bb-ycmf0)-xcmf0); caux=(bb-ycmf0)^2+xcmf0^2-Rmf^2;
yutemf0=yumf0;
xutemf0=(yutemf0-bb)/aa;
xutemf=xutemf0*Cos(amf)+yutemf0*Sin(amf); yutemf=-xutemf0*Sin(amf)+yutemf0*Cos(amf);
utemfP=newp; Point(utemfP)={xutemf,yutemf,0};
If (iujf == 0)
  uamfL=newl; Line(uamfL) = {ujmfP,utemfP};
  Else 
  //uamfL1=newl; Circle(uamfL1) = {ujmfP,cmfP,umfP};
  //uamfL2=newl; Line(uamfL2) = {umfP,utemfP};
  uamfL=newl; Line(uamfL) = {ujmfP,umfP,utemfP};
  //Compound Line(uamfL) = {uamfL1,uamfL2};
  //Compound Curve{uamfL1,uamfL2};
EndIf
utemfL=newl; Line(utemfL)={utenfP,utemfP};

// lower trailing edge intersect
aa=Tan(Atan(dycdxte)-Atan(dytdxte)+Pi/2);
bb=yte0-aa*xte0;
aaux=1+aa^2; baux=2*(aa*(bb-ycmf0)-xcmf0); caux=(bb-ycmf0)^2+xcmf0^2-Rmf^2;
yltemf0=ylmf0;
xltemf0=(yltemf0-bb)/aa;
xltemf=xltemf0*Cos(amf)+yltemf0*Sin(amf); yltemf=-xltemf0*Sin(amf)+yltemf0*Cos(amf);
ltemfP=newp; Point(ltemfP)={xltemf,yltemf,0};
If (iljf == 0)
  lamfL=newl; Line(lamfL) = {ljmfP,ltemfP};
  Else 
  //lamfL1=newl; Circle(lamfL1) = {ljmfP,cmfP,lmfP};
  //lamfL2=newl; Line(lamfL2) = {lmfP,ltemfP};
  lamfL=newl; Line(lamfL) = {ljmfP,lmfP,ltemfP};
  //Compound Line(lamfL) = {lamfL1,lamfL2};
  //Compound Curve{lamfL1,lamfL2};
EndIf
ltemfL=newl; Line(ltemfL)={ltenfP,ltemfP};

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* FARFIELD
aff=0;

// semicircle definition
xuff0=0; yuff0=a;
uff0P=newp; Point(uff0P)={xuff0,yuff0,0};

//aa=Tan(Atan(dycdxte)+Atan(dytdxte)+Pi/2);
//bb=yte0-aa*xte0;
//yuff=a; xuff=(yuff-bb)/aa;
//uffP=newp; Point(uffP)={xuff,yuff,0};

xlff0=0; ylff0=-a;
lff0P=newp; Point(lff0P)={xlff0,ylff0,0};

//aa=Tan(Atan(dycdxte)-Atan(dytdxte)+Pi/2);
//bb=yte0-aa*xte0;
//ylff=-a; xlff=(ylff-bb)/aa;
//lffP=newp; Point(lffP)={xlff,ylff,0};

// leading edge intersect
aa=Tan(Atan(dycdxle)+Pi/2+Pi/2);
bb=yle0-aa*xle0;
aaux=1+aa^2; baux=2*aa*bb; caux=bb^2-a^2;
xleff=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
yleff=aa*xleff+bb;
leffP=newp; Point(leffP)={xleff,yleff,0};
leffL=newl; Line(leffL)={lemfP,leffP};

// upper jet intersect
aa=Tan(ajint);//Tan(Atan(dycdxj)+Atan(dytdxj)+Pi/2);
bb=yuj0-aa*xuj0;
aaux=1+aa^2; baux=2*aa*bb; caux=bb^2-a^2;
xujff0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
If (xujff0 < xuff0)
  yujff0=aa*xujff0+bb;
  xujff=xujff0*Cos(aff)+yujff0*Sin(aff); yujff=-xujff0*Sin(aff)+yujff0*Cos(aff);
  ujffP=newp; Point(ujffP)={xujff,yujff,0};
  ufffL=newl; Circle(ufffL) = {leffP,teP,ujffP};
  iujaff=1;
  Else
  yujff0=yuff0;
  xujff0=(yujff0-bb)/aa;
  xujff=xujff0*Cos(aff)+yujff0*Sin(aff); yujff=-xujff0*Sin(aff)+yujff0*Cos(aff);
  ujffP=newp; Point(ujffP)={xujff,yujff,0};
  ufffL1=newl; Circle(ufffL1) = {leffP,teP,uff0P};
  ufffL2=newl; Line(ufffL2) = {uff0P,ujffP};
  ufffL=newl; Compound Line(ufffL) = {ufffL1,ufffL2};
  iujaff=0;
EndIf
ujffL=newl; Line(ujffL)={ujmfP,ujffP};

// lower jet intersect
aa=Tan(-ajint);//Tan(Atan(dycdxj)-Atan(dytdxj)+Pi/2);
bb=ylj0-aa*xlj0;
aaux=1+aa^2; baux=2*aa*bb; caux=bb^2-a^2;
xljff0=(-baux-Sqrt(baux^2-4*aaux*caux))/(2*aaux);
If (xljff0 < xlff0)
  yljff0=aa*xljff0+bb;
  xljff=xljff0*Cos(aff)+yljff0*Sin(aff); yljff=-xljff0*Sin(aff)+yljff0*Cos(aff);
  ljffP=newp; Point(ljffP)={xljff,yljff,0};
  lfffL=newl; Circle(lfffL) = {leffP,teP,ljffP};
  iljaff=1;
  Else
  yljff0=ylff0;
  xljff0=(yljff0-bb)/aa;
  xljff=xljff0*Cos(aff)+yljff0*Sin(aff); yljff=-xljff0*Sin(aff)+yljff0*Cos(aff);
  ljffP=newp; Point(ljff)={xljff,yljff,0};
  lfffL1=newl; Circle(lfffL1) = {leffP,teP,lff0P};
  lfffL2=newl; Line(lfffL2) = {lff0P,ljffP};
  lfffL=newl; Compound Line(lfffL) = {lfffL1,lfffL2};
  iljaff=0;
EndIf
ljffL=newl; Line(ljffL)={ljmfP,ljffP};

// upper trailing edge intersect
//aa=Tan(Atan(dycdxte)+Atan(dytdxte)+Pi/2);
//bb=yte0-aa*xte0;
//aaux=1+aa^2; baux=2*aa*bb; caux=bb^2-a^2;
//xuteff0=(-baux+Sqrt(baux^2-4*aaux*caux))/(2*aaux);
//Printf('%g',bb);
//If (xuteff0 < xuff0)
//  yuteff0=aa*xuteff0+bb;
//  xuteff=xuteff0*Cos(aff)+yuteff0*Sin(aff); yuteff=-xuteff0*Sin(aff)+yuteff0*Cos(aff);
//  uteffP=newp; Point(uteffP)={xuteff,yuteff,0};
//  uaffL=newl; Circle(uaffL) = {leffP,teP,uteffP};
//  iutefff=1;
//  Else
//  yuteff0=yuff0;
//  xuteff0=(yuteff0-bb)/aa;
//  xuteff=xuteff0*Cos(aff)+yuteff0*Sin(aff); yuteff=-xuteff0*Sin(aff)+yuteff0*Cos(aff);
//  uteffP=newp; Point(uteffP)={xuteff,yuteff,0};
//  If (iujaff == 1)
//    uaffL1=newl; Circle(uaffL1) = {ujffP,teP,uff0P};
//    uaffL2=newl; Line(uaffL2) = {uff0P,uteffP};
//    uaffL=newl; Compound Line(uaffL) = {uaffL1,uaffL2};
//    Else
//    uaffL=newl; Line(uaffL) = {ujffP,uteffP};
//  EndIf
//  iutefff=0;
//EndIf
//uteffL=newl; Line(uteffL)={utemfP,uteffP};
xuteff=xuff0; yuteff=yuff0;
uteffP=newp; Point(uteffP)={xuteff,yuteff,0};
uaffL=newl; Circle(uaffL) = {ujffP,teP,uteffP};
uteffL=newl; Line(uteffL)={utemfP,uteffP};


// lower trailing edge intersect
//aa=Tan(Atan(dycdxte)-Atan(dytdxte)+Pi/2);
//bb=yte0-aa*xte0;
//aaux=1+aa^2; baux=2*aa*bb; caux=bb^2-a^2;
//xlteff0=(-baux+Sqrt(baux^2-4*aaux*caux))/(2*aaux);
//If (xlteff0 < xlff0)
//  ylteff0=aa*xlteff0+bb;
//  xlteff=xlteff0*Cos(aff)+ylteff0*Sin(aff); ylteff=-xlteff0*Sin(aff)+ylteff0*Cos(aff);
//  lteffP=newp; Point(lteffP)={xlteff,ylteff,0};
//  laffL=newl; Circle(laffL) = {ljffP,teP,lteffP};
//  iltefff=1;
//  Else
//  ylteff0=ylff0;
//  xlteff0=(ylteff0-bb)/aa;
//  xlteff=xlteff0*Cos(aff)+ylteff0*Sin(aff); ylteff=-xlteff0*Sin(aff)+ylteff0*Cos(aff);
//  lteffP=newp; Point(lteffP)={xlteff,ylteff,0};
//  If (iujaff == 1)
//    laffL1=newl; Circle(laffL1) = {ljffP,teP,lff0P};
//    laffL2=newl; Line(laffL2) = {lff0P,lteffP};
//    laffL=newl; Compound Line(laffL) = {laffL1,laffL2};
//    Else
//    laffL=newl; Line(laffL) = {ljffP,lteffP};
//  EndIf
//  iltefff=0;
//EndIf
//lteffL=newl; Line(lteffL)={ltemfP,lteffP};
xlteff=xlff0; ylteff=ylff0;
lteffP=newp; Point(lteffP)={xlteff,ylteff,0};
laffL=newl; Circle(laffL) = {ljffP,teP,lteffP};
lteffL=newl; Line(lteffL)={ltemfP,lteffP};

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* DOWNSTREAM REGION

// Downstream points
xudff=b; yudff=a;
udffP=newp; Point(udffP)={xudff,yudff,0};
xldff=b; yldff=-a;
ldffP=newp; Point(ldffP)={xldff,yldff,0};

xudnf=b; yudnf=Tan(daoau)*xudnf+yutenf0;
udnfP=newp; Point(udnfP)={xudnf,yudnf,0};
xudmf=b; yudmf=Tan(daoauu)*xudmf+yutemf0;
udmfP=newp; Point(udmfP)={xudmf,yudmf,0};

xldnf=b; yldnf=-Tan(daoal)*xldnf+yltenf0;
ldnfP=newp; Point(ldnfP)={xldnf,yldnf,0};
xldmf=b; yldmf=-Tan(daoall)*xldmf+yltemf0;
ldmfP=newp; Point(ldmfP)={xldmf,yldmf,0};

//xwd=b; ywd=0;
xwd=b; ywd=(yudnf+yldnf)/2;
wdP=newp; Point(wdP)={xwd,ywd,0};

// horizontal lines
wL=newl; Line(wL)={teP,wdP};
uwnfL=newl; Line(uwnfL)={utenfP,udnfP};
uwmfL=newl; Line(uwmfL)={utemfP,udmfP};
uwffL=newl; Line(uwffL) = {uteffP,udffP};
lwnfL=newl; Line(lwnfL)={ltenfP,ldnfP};
lwmfL=newl; Line(lwmfL)={ltemfP,ldmfP};
lwffL=newl; Line(lwffL) = {lteffP,ldffP};

// vertical lines
udnfL=newl; Line(udnfL)={wdP,udnfP};
udmfL=newl; Line(udmfL)={udnfP,udmfP};
udffL=newl; Line(udffL) = {udmfP,udffP};
ldnfL=newl; Line(ldnfL)={wdP,ldnfP};
ldmfL=newl; Line(ldmfL)={ldnfP,ldmfP};
ldffL=newl; Line(ldffL) = {ldmfP,ldffP};

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* FRINGE REGION

If (fr>0)

  // Fringe points
  xuiff=xudff+fr; yuiff=yudff;
  uiffP=newp; Point(uiffP)={xuiff,yuiff,0};
  xliff=xldff+fr; yliff=yldff;
  liffP=newp; Point(liffP)={xliff,yliff,0};

  xuinf=xudnf+fr; yuinf=yudnf;
  uinfP=newp; Point(uinfP)={xuinf,yuinf,0};
  xuimf=xudmf+fr; yuimf=yudmf;
  uimfP=newp; Point(uimfP)={xuimf,yuimf,0};

  xlinf=xldnf+fr; ylinf=yldnf;
  linfP=newp; Point(linfP)={xlinf,ylinf,0};
  xlimf=xldmf+fr; ylimf=yldmf;
  limfP=newp; Point(limfP)={xlimf,ylimf,0};

  xwi=xwd+fr; ywi=ywd;
  wiP=newp; Point(wiP)={xwi,ywi,0};

  // horizontal lines
  wiL=newl; Line(wiL)={wdP,wiP};
  uwinfL=newl; Line(uwinfL)={udnfP,uinfP};
  uwimfL=newl; Line(uwimfL)={udmfP,uimfP};
  uwiffL=newl; Line(uwiffL) = {udffP,uiffP};
  lwinfL=newl; Line(lwinfL)={ldnfP,linfP};
  lwimfL=newl; Line(lwimfL)={ldmfP,limfP};
  lwiffL=newl; Line(lwiffL) = {ldffP,liffP};

  // vertical lines
  uinfL=newl; Line(uinfL)={wiP,uinfP};
  uimfL=newl; Line(uimfL)={uinfP,uimfP};
  uiffL=newl; Line(uiffL) = {uimfP,uiffP};
  linfL=newl; Line(linfL)={wiP,linfP};
  limfL=newl; Line(limfL)={linfP,limfP};
  liffL=newl; Line(liffL) = {limfP,liffP};

EndIf


//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* * * SET MESH
//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//

//* along airfoil
Luas=c*(1-pj); // approx
LNuas=Luas/(Nua-1);//*kbuas^0.5;
Printf('LNuas= %g',LNuas);

//* first node correspondances
xfirst[0]=0;
xfirst[1]=1;
xfirst[2]=1;
xfirst[3]=0.5;
xfirst[4]=0.211324865405187;
xfirst[5]=0.112701665379258;
xfirst[6]=0.0694318442029737;
xfirst[7]=0.0469100770306681;
xfirst[8]=0.033765242898424;
xfirst[9]=0.0254460438286208;
xfirst[10]=0.019855071751232;
xfirst[11]=0.0159198802461871;
xfirst[12]=0.0130467357414141;
xfirst[13]=0.0108856709269715;
xfirst[14]=0.00921968287664043;
xfirst[15]=0.00790847264070593;
xfirst[16]=0.00685809565159379;
xfirst[17]=0.00600374098975753;

//* * * NEARFIELD

//* upper trailing edge nearfield
Putenf=Puanf;
Lfirstu=xfirstu/xfirst[Putenf];
l1utenf=Lfirstu*(1-kutenf)/(1-kutenf^(Putenf-1));
Printf('%g',kutenf);
//l1l=1e-3; // adjust to fit virtual y+ on lower surface
Lutenf=Sqrt((xutenf-xte)^2+(yutenf-yte)^2); // total lower nearfield thickness
Kutenf=kutenf^(Putenf-1); // bundle progression ratio
L1utenf=l1utenf*(1-kutenf^(Putenf-1))/(1-kutenf); // first bundle size
Nnf=Ceil(1+Log(1-Lutenf*(1-Kutenf)/L1utenf)/Log(Kutenf)); // Point count (rounded up)
nnf=(Nnf-1)*(Putenf-1)+1; // Virtual point count
If (ikeepl1k==0) //* Option 1: recalc l1, L1 for rounded Nnf
  L1utenf=Lutenf*(Kutenf-1)/(Kutenf^(Nnf-1)-1); // recalc L1 (for integer n)
  l1utenf=L1utenf*(1-kutenf)/(1-kutenf^(Putenf-1)); // recalc l1 (for integer n)
Else //* Option 2: recalc kutenf, Kutenf for rounded Nnf
  lt=Lutenf; l1=L1utenf; n=Nnf; kguess=Kutenf;
  Call Compk;
  Kutenf=kguess;
  kutenf=Kutenf^(1/(Putenf-1)); // progression ratio of bundle
  l1utenf=L1utenf*(1-kutenf)/(1-kutenf^(Putenf-1)); // first bundle size
EndIf
Nnfc=1+Log(1-Lutenf*(1-Kutenf)/L1utenf)/Log(Kutenf); // Point count (rounded up) check!
L1utenfc=Lutenf*(Kutenf-1)/(Kutenf^(Nnf-1)-1); // recalc L1 (= target l1?) check!
l1utenfc=L1utenfc*(1-kutenf)/(1-kutenf^(Putenf-1)); // recalc l1 (= target l1?) check!
LNutenf=L1utenf*Kutenf^(Nnf-2); // Last interval size
lnutenf=LNutenf*(1-kutenf)/(1-kutenf^(Putenf-1))*kutenf^(Putenf-2) ; // last virtual interval size
Printf('upper trailing edge nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lutenf,l1utenf,lnutenf,nnf,kutenf);
Printf('upper trailing edge nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lutenf,L1utenf,LNutenf,Nnf,Kutenf);
Printf('upper trailing edge nearfield checks: l1=%g; L1=%g;N=%g',l1utenfc,L1utenfc,Nnfc);
Printf('upper trailing edge nearfield: y1= %g',L1utenf*xfirst[Putenf]);

//* lower trailing edge nearfield
Pltenf=Planf;
Lfirstl=xfirstl/xfirst[Pltenf];
L1ltenf=Lfirstl;
Lltenf=Sqrt((xltenf-xte)^2+(yltenf-yte)^2);
nlte=(Nnf-1)*(Pltenf-1)+1;
// approx
l1ltenf=l1utenf; // first interval size (1st point height from airfoil at te)
lt=Lltenf; l1=l1ltenf; n=nlte; kguess=2.2; // target l1 (new order) MODIFIED KGUESS!
Call Compk;
kltenf=kguess;
Kltenf=kltenf^(Pltenf-1);
// exact
lt=Lltenf; l1=L1ltenf; n=Nnf; kguess=Kltenf;
Call Compk;
Kltenf=kguess;
kltenf=Kltenf^(1/(Pltenf-1)); // progression ratio of bundle
l1ltenf=L1ltenf*(1-kltenf)/(1-kltenf^(Pltenf-1)); // first bundle size
Nnfc=1+Log(1-Lltenf*(1-Kltenf)/L1ltenf)/Log(Kltenf); // Point count (rounded up) check!
L1ltenfc=Lltenf*(Kltenf-1)/(Kltenf^(Nnf-1)-1); // recalc L1 (= target l1?) check!
l1ltenfc=L1ltenfc*(1-kltenf)/(1-kltenf^(Pltenf-1)); // recalc l1 (= target l1?) check!
LNltenf=L1ltenf*Kltenf^(Nnf-2); // Last interval size
lnltenf=LNltenf*(1-kltenf)/(1-kltenf^(Pltenf-1))*kltenf^(Pltenf-2) ; // last virtual interval size
Printf('lower trailing edge nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lltenf,l1ltenf,lnltenf,nnf,kltenf);
Printf('lower trailing edge nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lltenf,L1ltenf,LNltenf,Nnf,Kltenf);
Printf('lower trailing edge nearfield checks: l1=%g; L1=%g;N=%g',l1ltenfc,L1ltenfc,Nnfc);
Printf('lower trailing edge nearfield: y1= %g',L1ltenf*xfirst[Pltenf]);


//* upper jet nearfield
Puj=Puanf;
Lfirstuj=xfirstuj/xfirst[Puj];
L1uj=Lfirstuj;
Luj=Sqrt((xujnf-xuj)^2+(yujnf-yuj)^2);
nujnf=(Nnf-1)*(Puj-1)+1;
// approx
l1uj=l1utenf*xfirstuj/xfirstu;
lt=Luj; l1=l1uj; n=nujnf; kguess=-1;
Call Compk;
kujnf=kguess;
Kujnf=kujnf^(Puj-1); // progression ratio of bundle
// exact
lt=Luj; l1=L1uj; n=Nnf; kguess=Kujnf;
Call Compk;
Kujnf=kguess;
kujnf=Kujnf^(1/(Puj-1)); // progression ratio of bundle
l1uj=L1uj*(1-kujnf)/(1-kujnf^(Puj-1)); // first bundle size
Nnfc=1+Log(1-Luj*(1-Kujnf)/L1uj)/Log(Kujnf); // Point count (rounded up) check!
L1ujc=Luj*(Kujnf-1)/(Kujnf^(Nnf-1)-1); // recalc L1 (= target l1?) check!
l1ujc=L1uj*(1-kujnf)/(1-kujnf^(Puj-1)); // recalc l1 (= target l1?) check!
LNuj=L1uj*Kujnf^(Nnf-2); // Last interval size
lnuj=LNuj*(1-kujnf)/(1-kujnf^(Puj-1))*kujnf^(Puj-2) ; // last virtual interval size
Printf('upper jet nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Luj,l1uj,lnuj,nujnf,kujnf);
Printf('upper jet nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Luj,L1uj,LNuj,Nnf,Kujnf);
Printf('upper jet nearfield checks: l1=%g; L1=%g;N=%g',l1uj,L1uj,Nnfc);
Printf('upper jet nearfield: y1= %g',L1uj*xfirst[Puj]);

//* lower jet nearfield
Plj=Planf;
Lfirstlj=xfirstlj/xfirst[Plj];
L1lj=Lfirstlj;
Llj=Sqrt((xljnf-xlj)^2+(yljnf-ylj)^2);
nljnf=(Nnf-1)*(Plj-1)+1;
// approx
l1lj=l1utenf*xfirstlj/xfirstu;
//kljnf=kltenf; Kljnf=Kltenf; 
lt=Llj; l1=l1lj; n=nljnf; kguess=-1;
Call Compk;
kljnf=kguess;
Kljnf=kljnf^(Plj-1); // progression ratio of bundle
// exact
lt=Llj; l1=L1lj; n=Nnf; kguess=Kljnf;
Call Compk;
Kljnf=kguess;
kljnf=Kljnf^(1/(Plj-1)); // progression ratio of bundle
l1lj=L1lj*(1-kljnf)/(1-kljnf^(Plj-1)); // first bundle size
Nnfc=1+Log(1-Llj*(1-Kljnf)/L1lj)/Log(Kljnf); // Point count (rounded up) check!
L1ljc=Llj*(Kljnf-1)/(Kljnf^(Nnf-1)-1); // recalc L1 (= target l1?) check!
l1ljc=L1lj*(1-kljnf)/(1-kljnf^(Plj-1)); // recalc l1 (= target l1?) check!
LNlj=L1lj*Kljnf^(Nnf-2); // Last interval size
lnlj=LNlj*(1-kljnf)/(1-kljnf^(Plj-1))*kljnf^(Plj-2) ; // last virtual interval size
Printf('lower jet nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Llj,l1lj,lnlj,nljnf,kljnf);
Printf('lower jet nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Llj,L1lj,LNlj,Nnf,Kljnf);
Printf('lower jet nearfield checks: l1=%g; L1=%g;N=%g',l1lj,L1lj,Nnfc);
Printf('lower jet nearfield: y1= %g',L1lj*xfirst[Plj]);


//* leading edge nearfield
Ple=Pufnf;
Lfirstle=xfirstle/xfirst[Ple];
L1le=Lfirstle;
Lle=Sqrt((xlenf-xle)^2+(ylenf-yle)^2);
nlenf=(Nnf-1)*(Ple-1)+1;
// approx
l1le=l1utenf*xfirstle/xfirstu;
//klenf=kfl; Klenf=Kfl; 
lt=Lle; l1=l1le; n=nlenf; kguess=-1;
Call Compk;
klenf=kguess;
Klenf=klenf^(Ple-1); // progression ratio of bundle
// exact
lt=Lle; l1=L1le; n=Nnf; kguess=Klenf;
Call Compk;
Klenf=kguess;
klenf=Klenf^(1/(Ple-1)); // progression ratio of bundle
l1le=L1le*(1-klenf)/(1-klenf^(Ple-1)); // first bundle size
Nnfc=1+Log(1-Lle*(1-Klenf)/L1le)/Log(Klenf); // Point count (rounded up) check!
L1lec=Lle*(Klenf-1)/(Klenf^(Nnf-1)-1); // recalc L1 (= target l1?) check!
l1lec=L1le*(1-klenf)/(1-klenf^(Ple-1)); // recalc l1 (= target l1?) check!
LNle=L1le*Klenf^(Nnf-2); // Last interval size
lnle=LNle*(1-klenf)/(1-klenf^(Ple-1))*klenf^(Ple-2) ; // last virtual interval size
Printf('leading edge nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lle,l1le,lnle,nlenf,klenf);
Printf('leading edge nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lle,L1le,LNle,Nnf,Klenf);
Printf('leading edge nearfield checks: l1=%g; L1=%g;N=%g',l1le,L1le,Nnfc);
Printf('leading edge nearfield: y1= %g',L1le*xfirst[Ple]);


//* * * MIDFIELD

//* upper trailing edge midfield
Putemf=Puamf;
Lutemf=Sqrt((xutemf-xutenf)^2+(yutemf-yutenf)^2);
Kutemf=kutemf^(Putemf-1);
L1utemf=LNutenf; // l1 mf inherited from ln nf
//L1utemf=LNutenf*Kutenf; // l1 mf inherited from ln nf
l1utemf=L1utemf*(1-kutemf)/(1-kutemf^(Putemf-1)); // first bundle size
Nmf=Ceil(1+Log(1-Lutemf*(1-Kutemf)/L1utemf)/Log(Kutemf)); // Point count (rounded up)
//Printf('%g %g %g %g %g %g %g',kutemf,Kutemf,l1utemf,L1utemf,Lutemf,1+Log(1-Lutemf*(1-Kutemf)/L1utemf)/Log(Kutemf),Nmf);
nmf=(Nmf-1)*(Putemf-1)+1; // Virtual point count
If (ikeepl1k==0) //* Option 1: recalc l1, L1 for rounded Nmf
  L1utemf=Lutemf*(Kutemf-1)/(Kutemf^(Nmf-1)-1); // recalc L1 (for integer n)
  l1utemf=L1utemf*(1-kutemf)/(1-kutemf^(Putemf-1)); // recalc l1 (for integer n)
Else //* Option 2: recalc kutemf, Kutemf for rounded Nmf
  lt=Lutemf; l1=L1utemf; n=Nmf; kguess=Kutemf;
  Call Compk;
  Kutemf=kguess;
  kutemf=Kutemf^(1/(Putemf-1)); // progression ratio of bundle
  l1utemf=L1utemf*(1-kutemf)/(1-kutemf^(Putemf-1)); // first bundle size
EndIf
LNmf=L1utemf*Kutemf^(Nmf-2); // Last bundle size
lnmf=LNmf*(1-kutemf)/(1-kutemf^(Putemf-1))*kutemf^(Putemf-2) ; // last virtual interval size
Printf('upper trailing edge midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lutemf,l1utemf,lnmf,nmf,kutemf);
Printf('upper trailing edge midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lutemf,L1utemf,LNmf,Nmf,Kutemf);

//* upper jet midfield
Pujmf=Puamf;
Lujmf=Sqrt((xujmf-xujnf)^2+(yujmf-yujnf)^2);
//If (Kujnf>1)
//  L1ujmf=LNuj*Kujnf;
//  Else
  L1ujmf=LNuj; // l1 mf inherited from ln nf
//EndIf
lt=Lujmf; l1=L1ujmf; n=Nmf; kguess=-1;
Call Compk;
Kujmf=kguess;
kujmf=Kujmf^(1./(Pujmf-1)); // progression ratio of bundle
l1ujmf=L1ujmf*(1-kujmf)/(1-kujmf^(Pujmf-1)); // first bundle size
Nmfc=1+Log(1-Lujmf*(1-Kujmf)/L1ujmf)/Log(Kujmf); // Point count (rounded up) check!
L1ujmfc=Lujmf*(Kujmf-1)/(Kujmf^(Nmf-1)-1); // recalc L1 (= target l1?) check!
l1ujmfc=L1ujmfc*(1-kujmf)/(1-kujmf^(Pujmf-1)); // recalc l1 (= target l1?) check!
LNujmf=L1ujmf*Kujmf^(Nmf-2); // Last interval size
lnujmf=LNujmf*(1-kujmf)/(1-kujmf^(Pujmf-1))*kujmf^(Pujmf-2) ; // last virtual interval size
Printf('upper jet midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lujmf,l1ujmf,lnujmf,nmf,kujmf);
Printf('upper jet midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lujmf,L1ujmf,LNujmf,Nmf,Kujmf);
Printf('upper jet midfield checks: l1=%g; L1=%g;N=%g',l1ujmfc,L1ujmfc,Nmfc);

//* lower trailing edge midfield
//Plamf=Plamf;
Lltemf=Sqrt((xltemf-xltenf)^2+(yltemf-yltenf)^2);
//If (Kltenf>1)
//  L1ltemf=LNltenf*Kltenf;
//  Else
  L1ltemf=LNltenf; // l1 ff inherited from ln nf
//EndIf
lt=Lltemf; l1=L1ltemf; n=Nmf; kguess=-1;
Call Compk;
Kltemf=kguess;
kltemf=Kltemf^(1./(Plamf-1)); // progression ratio of bundle
l1ltemf=L1ltemf*(1-kltemf)/(1-kltemf^(Plamf-1)); // first bundle size
Nmfc=1+Log(1-Lltemf*(1-Kltemf)/L1ltemf)/Log(Kltemf); // Point count (rounded up) check!
L1ltemfc=Lltemf*(Kltemf-1)/(Kltemf^(Nmf-1)-1); // recalc L1 (= target l1?) check!
l1ltemfc=L1ltemfc*(1-kltemf)/(1-kltemf^(Plamf-1)); // recalc l1 (= target l1?) check!
LNltemf=L1ltemf*Kltemf^(Nmf-2); // Last interval size
lnltemf=LNltemf*(1-kltemf)/(1-kltemf^(Plamf-1))*kltemf^(Plamf-2) ; // last virtual interval size
Printf('lower trailing edge midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lltemf,l1ltemf,lnltemf,nmf,kltemf);
Printf('lower trailing edge midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lltemf,L1ltemf,LNltemf,Nmf,Kltemf);
Printf('lower trailing edge midfield checks: l1=%g; L1=%g;N=%g',l1ltemfc,L1ltemfc,Nmfc);

//* lower jet midfield
Pljmf=Plamf;
Lljmf=Sqrt((xljmf-xljnf)^2+(yljmf-yljnf)^2);
//If (Kljnf>1.0)
//  L1ljmf=LNlj*Kljnf;
//  Else
  L1ljmf=LNlj; // l1 mf inherited from ln nf
//EndIf
lt=Lljmf; l1=L1ljmf; n=Nmf; kguess=-1;
Call Compk;
Kljmf=kguess;
kljmf=Kljmf^(1./(Pljmf-1)); // progression ratio of bundle
l1ljmf=L1ljmf*(1-kljmf)/(1-kljmf^(Pljmf-1)); // first bundle size
Nmfc=1+Log(1-Lljmf*(1-Kljmf)/L1ljmf)/Log(Kljmf); // Point count (rounded up) check!
L1ljmfc=Lljmf*(Kljmf-1)/(Kljmf^(Nmf-1)-1); // recalc L1 (= target l1?) check!
l1ljmfc=L1ljmfc*(1-kljmf)/(1-kljmf^(Pljmf-1)); // recalc l1 (= target l1?) check!
LNljmf=L1ljmf*Kljmf^(Nmf-2); // Last interval size
lnljmf=LNljmf*(1-kljmf)/(1-kljmf^(Pljmf-1))*kljmf^(Pljmf-2) ; // last virtual interval size
Printf('lower jet midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lljmf,l1ljmf,lnljmf,nmf,kljmf);
Printf('lower jet midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lljmf,L1ljmf,LNljmf,Nmf,Kljmf);
Printf('lower jet midfield checks: l1=%g; L1=%g;N=%g',l1ljmfc,L1ljmfc,Nmfc);

//* leading edge midfield
Plemf=Plamf;
Llemf=Sqrt((xlemf-xlenf)^2+(ylemf-ylenf)^2);
//If (Klenf>1)
//  L1lemf=LNle*Klenf;
//  Else
  L1lemf=LNle; // l1 mf inherited from ln nf
//EndIf
lt=Llemf; l1=L1lemf; n=Nmf; kguess=-1;
Call Compk;
Klemf=kguess;
klemf=Klemf^(1./(Plemf-1)); // progression ratio of bundle
l1lemf=L1lemf*(1-klemf)/(1-klemf^(Plemf-1)); // first bundle size
Nmfc=1+Log(1-Llemf*(1-Klemf)/L1lemf)/Log(Klemf); // Point count (rounded up) check!
L1lemfc=Llemf*(Klemf-1)/(Klemf^(Nmf-1)-1); // recalc L1 (= target l1?) check!
l1lemfc=L1lemfc*(1-klemf)/(1-klemf^(Plemf-1)); // recalc l1 (= target l1?) check!
LNlemf=L1lemf*Klemf^(Nmf-2); // Last interval size
lnlemf=LNlemf*(1-klemf)/(1-klemf^(Plemf-1))*klemf^(Plemf-2) ; // last virtual interval size
Printf('leading edge midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Llemf,l1lemf,lnlemf,nmf,klemf);
Printf('leading edge midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Llemf,L1lemf,LNlemf,Nmf,Klemf);
Printf('leading edge midfield checks: l1=%g; L1=%g;N=%g',l1lemfc,L1lemfc,Nmfc);


//* * * FARFIELD

//* upper trailing edge farfield
Puteff=Pff;
Luteff=Sqrt((xuteff-xutemf)^2+(yuteff-yutemf)^2);
Kuteff=kuteff^(Pff-1);
L1uteff=LNmf; // l1 ff inherited from ln mf
//L1uteff=LNmf*Kutemf; // l1 ff inherited from ln mf
l1uteff=L1uteff*(1-kuteff)/(1-kuteff^(Puteff-1)); // first bundle size
Nff=Ceil(1+Log(1-Luteff*(1-Kuteff)/L1uteff)/Log(Kuteff)); // Point count (rounded up)
nff=(Nff-1)*(Puteff-1)+1; // Virtual point count
If (ikeepl1k==0) //* Option 1: recalc l1, L1 for rounded Nff
  L1uteff=Luteff*(Kuteff-1)/(Kuteff^(Nff-1)-1); // recalc L1 (for integer n)
  l1uteff=L1uteff*(1-kuteff)/(1-kuteff^(Puteff-1)); // recalc l1 (for integer n)
Else //* Option 2: recalc kuteff, Kuteff for rounded Nff
  lt=Luteff; l1=L1uteff; n=Nff; kguess=Kuteff;
  Call Compk;
  Kuteff=kguess;
  kuteff=Kuteff^(1/(Puteff-1)); // progression ratio of bundle
  l1uteff=L1uteff*(1-kuteff)/(1-kuteff^(Puteff-1)); // first bundle size
EndIf
LNuteff=L1uteff*Kuteff^(Nff-2); // Last bundle size
lnuteff=LNuteff*(1-kuteff)/(1-kuteff^(Puteff-1))*kuteff^(Puteff-2) ; // last virtual interval size
Printf('upper trailing edge farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Luteff,l1uteff,lnuteff,nff,kuteff);
Printf('upper trailing edge farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Luteff,L1uteff,LNuteff,Nff,Kuteff);

//* upper jet farfield
Pujff=Pff;
Lujff=Sqrt((xujff-xujmf)^2+(yujff-yujmf)^2);
L1ujff=LNujmf; // l1 ff inherited from ln mf
//L1ujff=LNujmf*Kujmf; // l1 ff inherited from ln mf
lt=Lujff; l1=L1ujff; n=Nff; kguess=3;
Call Compk;
Kujff=kguess;
kujff=Kujff^(1./(Pujff-1)); // progression ratio of bundle
l1ujff=L1ujff*(1-kujff)/(1-kujff^(Pujff-1)); // first bundle size
Nffc=1+Log(1-Lujff*(1-Kujff)/L1ujff)/Log(Kujff); // Point count (rounded up) check!
L1ujffc=Lujff*(Kujff-1)/(Kujff^(Nff-1)-1); // recalc L1 (= target l1?) check!
l1ujffc=L1ujffc*(1-kujff)/(1-kujff^(Pujff-1)); // recalc l1 (= target l1?) check!
LNujff=L1ujff*Kujff^(Nff-2); // Last interval size
lnujff=LNujff*(1-kujff)/(1-kujff^(Pujff-1))*kujff^(Pujff-2) ; // last virtual interval size
Printf('upper jet farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lujff,l1ujff,lnujff,nff,kujff);
Printf('upper jet farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lujff,L1ujff,LNujff,Nff,Kujff);
Printf('upper jet farfield checks: l1=%g; L1=%g;N=%g',l1ujffc,L1ujffc,Nffc);

//* lower trailing edge farfield
Plteff=Pff;
Llteff=Sqrt((xlteff-xltemf)^2+(ylteff-yltemf)^2);
L1lteff=LNltemf; // l1 ff inherited from ln mf
//L1lteff=LNltemf*Kltemf; // l1 ff inherited from ln mf
lt=Llteff; l1=L1lteff; n=Nff; kguess=3;
Call Compk;
Klteff=kguess;
klteff=Klteff^(1./(Plteff-1)); // progression ratio of bundle
l1lteff=L1lteff*(1-klteff)/(1-klteff^(Plteff-1)); // first bundle size
Nffc=1+Log(1-Llteff*(1-Klteff)/L1lteff)/Log(Klteff); // Point count (rounded up) check!
L1lteffc=Llteff*(Klteff-1)/(Klteff^(Nff-1)-1); // recalc L1 (= target l1?) check!
l1lteffc=L1lteffc*(1-klteff)/(1-klteff^(Plteff-1)); // recalc l1 (= target l1?) check!
LNlteff=L1lteff*Klteff^(Nff-2); // Last interval size
lnlteff=LNlteff*(1-klteff)/(1-klteff^(Plteff-1))*klteff^(Plteff-2) ; // last virtual interval size
Printf('lower trailing edge farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Llteff,l1lteff,lnlteff,nff,klteff);
Printf('lower trailing edge farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Llteff,L1lteff,LNlteff,Nff,Klteff);
Printf('lower trailing edge farfield checks: l1=%g; L1=%g;N=%g',l1lteffc,L1lteffc,Nffc);

//* lower jet farfield
Pljff=Pff;
Lljff=Sqrt((xljff-xljmf)^2+(yljff-yljmf)^2);
L1ljff=LNljmf; // l1 ff inherited from ln mf
//L1ljff=LNljmf*Kljmf; // l1 ff inherited from ln mf
lt=Lljff; l1=L1ljff; n=Nff; kguess=3;
Call Compk;
Kljff=kguess;
kljff=Kljff^(1./(Pljff-1)); // progression ratio of bundle
l1ljff=L1ljff*(1-kljff)/(1-kljff^(Pljff-1)); // first bundle size
Nffc=1+Log(1-Lljff*(1-Kljff)/L1ljff)/Log(Kljff); // Point count (rounded up) check!
L1ljffc=Lljff*(Kljff-1)/(Kljff^(Nff-1)-1); // recalc L1 (= target l1?) check!
l1ljffc=L1ljffc*(1-kljff)/(1-kljff^(Pljff-1)); // recalc l1 (= target l1?) check!
LNljff=L1ljff*Kljff^(Nff-2); // Last interval size
lnljff=LNljff*(1-kljff)/(1-kljff^(Pljff-1))*kljff^(Pljff-2) ; // last virtual interval size
Printf('lower jet farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lljff,l1ljff,lnljff,nff,kljff);
Printf('lower jet farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lljff,L1ljff,LNljff,Nff,Kljff);
Printf('lower jet farfield checks: l1=%g; L1=%g;N=%g',l1ljffc,L1ljffc,Nffc);

//* leading edge farfield
Pleff=Pff;
Lleff=Sqrt((xleff-xlemf)^2+(yleff-ylemf)^2);
L1leff=LNlemf; // l1 ff inherited from ln mf
//L1leff=LNlemf*Klemf; // l1 ff inherited from ln mf
lt=Lleff; l1=L1leff; n=Nff; kguess=3;
Call Compk;
Kleff=kguess;
kleff=Kleff^(1./(Pleff-1)); // progression ratio of bundle
l1leff=L1leff*(1-kleff)/(1-kleff^(Pleff-1)); // first bundle size
Nffc=1+Log(1-Lleff*(1-Kleff)/L1leff)/Log(Kleff); // Point count (rounded up) check!
L1leffc=Lleff*(Kleff-1)/(Kleff^(Nff-1)-1); // recalc L1 (= target l1?) check!
l1leffc=L1leffc*(1-kleff)/(1-kleff^(Pleff-1)); // recalc l1 (= target l1?) check!
LNleff=L1leff*Kleff^(Nff-2); // Last interval size
lnleff=LNleff*(1-kleff)/(1-kleff^(Pleff-1))*kleff^(Pleff-2) ; // last virtual interval size
Printf('leading edge farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lleff,l1leff,lnleff,nff,kleff);
Printf('leading edge farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lleff,L1leff,LNleff,Nff,Kleff);
Printf('leading edge farfield checks: l1=%g; L1=%g;N=%g',l1leffc,L1leffc,Nffc);


//* * * DOWNSTREAM WAKE REGION

//* downstream wake
//l1w=l1l*5; // would be nice to use last bump cell on airfoil
L1w=LNuas; // first bundle size
Lw=Sqrt((xwd-xte)^2+(ywd-yte)^2);
Kw=kw^(Pw-1); // bundle progression ratio
l1w=L1w*(1-kw)/(1-kw^(Pw-1)); // first interval size
Nw=Ceil(1+Log(1-Lw*(1-Kw)/L1w)/Log(Kw)); // Point count (rounded up)
nw=(Nw-1)*(Pw-1)+1; // Virtual point count
L1w=Lw*(Kw-1)/(Kw^(Nw-1)-1); // recalc L1
l1w=L1w*(1-kw)/(1-kw^(Pw-1)); // recalc l1
LNw=L1w*Kw^(Nw-2); // Last bundle size
lnw=LNw*(1-kw)/(1-kw^(Pw-1))*kw^(Pw-2) ; // last virtual interval size
Printf('wake: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lw,l1w,lnw,nw,kw);
Printf('wake: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lw,L1w,LNw,Nw,Kw);

//* upper downstream wake nearfield
Pudnf=Pw;
Ludnf=Sqrt((xudnf-xwd)^2+(yudnf-ywd)^2);
Kudnf=kudnf^(Pudnf-1);
If (kudnf!=1.0)
  L1udnf=Ludnf*(Kudnf-1)/(Kudnf^(Nnf-1)-1); // recalc L1
  l1udnf=L1udnf*(1-kudnf)/(1-kudnf^(Pudnf-1)); // recalc l1
  LNudnf=L1udnf*Kudnf^(Nnf-2); // Last interval size
  lnudnf=LNudnf*(1-kudnf)/(1-kudnf^(Pudnf-1))*kudnf^(Pudnf-2); // last virtual interval size
Else
  L1udnf=Ludnf/(Nnf-1);
  l1udnf=Ludnf/(nnf-1);
  LNudnf=L1udnf;
  lnudnf=LNudnf;
EndIf
Printf('upper downstream wake nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Ludnf,l1udnf,lnudnf,nnf,kudnf);
Printf('upper downstream wake nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Ludnf,L1udnf,LNudnf,Nnf,Kudnf);

//* lower downstream wake nearfield
Pldnf=Pw;
Lldnf=Sqrt((xldnf-xwd)^2+(yldnf-ywd)^2);
//Printf('culo %g %g',yudnf,yldnf);
Kldnf=kldnf^(Pldnf-1);
If (kldnf!=1.0)
  L1ldnf=Lldnf*(Kldnf-1)/(Kldnf^(Nnf-1)-1); // recalc L1
  l1ldnf=L1ldnf*(1-kldnf)/(1-kldnf^(Pldnf-1)); // recalc l1
  LNldnf=L1ldnf*Kldnf^(Nnf-2); // Last interval size
  lnldnf=LNldnf*(1-kldnf)/(1-kldnf^(Pldnf-1))*kldnf^(Pldnf-2); // last virtual interval size
Else
  L1ldnf=Lldnf/(Nnf-1);
  l1ldnf=Lldnf/(nnf-1);
  LNldnf=L1ldnf;
  lnldnf=LNldnf;
EndIf
Printf('lower downstream wake nearfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lldnf,l1ldnf,lnldnf,nnf,kldnf);
Printf('lower downstream wake nearfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lldnf,L1ldnf,LNldnf,Nnf,Kldnf);

//* upper downstream wake midfield
Pudmf=Puwmf;
Ludmf=Sqrt((xudmf-xudnf)^2+(yudmf-yudnf)^2); // total length
L1udmf=Ludmf/(Nmf-1);
l1udmf=Ludmf/(nmf-1);
LNudmf=L1udmf;
lnudmf=LNudmf;
Kudmf=1.0;
//L1udmf=LNudnf*(Pudmf-1)/(Pudnf-1); // target l1 (keeping order)
//lt=Ludmf; l1=L1udmf; n=Nmf; kguess=-1; // target l1 (new order)
//Call Compk;
//Kudmf=kguess;
//kudmf=Kudmf^(1/(Pudmf-1));
//l1udmf=L1udmf*(1-kudmf)/(1-kudmf^(Pudmf-1)); // first bundle size
//Nmfc=1+Log(1-Ludmf*(1-Kudmf)/L1udmf)/Log(Kudmf); // Point count (rounded up) check!
//L1udmfc=Ludmf*(Kudmf-1)/(Kudmf^(Nmf-1)-1); // recalc L1 (= target l1?) check!
//l1udmfc=L1udmfc*(1-kudmf)/(1-kudmf^(Pudmf-1)); // recalc l1 (= target l1?) check!
//LNudmf=L1udmf*Kudmf^(Nmf-2); // Last interval size
//lnudmf=LNudmf*(1-kudmf)/(1-kudmf^(Pudmf-1))*kudmf^(Pudmf-2) ; // last virtual interval size
Printf('upper downstream wake midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Ludmf,l1udmf,lnudmf,nmf,kudmf);
Printf('upper downstream wake midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Ludmf,L1udmf,LNudmf,Nmf,Kudmf);
//Printf('upper midfield wake aft checks: l1=%g; L1=%g;N=%g',l1udmfc,L1udmfc,Nmfc);

//* lower downstream wake midfield
Pldmf=Plwmf;
Lldmf=Sqrt((xldmf-xldnf)^2+(yldmf-yldnf)^2); // total length
L1ldmf=Lldmf/(Nmf-1);
l1ldmf=Lldmf/(nmf-1);
LNldmf=L1ldmf;
lnldmf=LNldmf;
Kldmf=1.0; kldmf=1.0;
//L1ldmf=LNldnf*(Pldmf-1)/(Pldnf-1); // target L1 (keeping order)
//lt=Lldmf; l1=L1ldmf; n=Nmf; kguess=-1; // target l1 (new order)
//Call Compk;
//Kldmf=kguess;
//kldmf=Kldmf^(1/(Pldmf-1));
//l1ldmf=L1ldmf*(1-kldmf)/(1-kldmf^(Pldmf-1)); // first bundle size
//Nmfc=1+Log(1-Lldmf*(1-Kldmf)/L1ldmf)/Log(Kldmf); // Point count (rounded up) check!
//L1ldmfc=Lldmf*(Kldmf-1)/(Kldmf^(Nmf-1)-1); // recalc L1 (= target l1?) check!
//l1ldmfc=L1ldmfc*(1-kldmf)/(1-kldmf^(Pldmf-1)); // recalc l1 (= target l1?) check!
//LNldmf=L1ldmf*Kldmf^(Nmf-2); // Last interval size
//lnldmf=LNldmf*(1-kldmf)/(1-kldmf^(Pldmf-1))*kldmf^(Pldmf-2) ; // last virtual interval size
Printf('lower downstream wake midfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lldmf,l1ldmf,lnldmf,nmf,kldmf);
Printf('lower downstream wake midfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lldmf,L1ldmf,LNldmf,Nmf,Kldmf);
//Printf('lower midfield wake aft checks: l1=%g; L1=%g;N=%g',l1ldmfc,L1ldmfc,Nmfc);

//* upper downstream wake farfield
Pudff=Pff;
Ludff=Sqrt((xudff-xudmf)^2+(yudff-yudmf)^2); // total length
L1udff=LNudmf*(Pudff-1)/(Pudmf-1); // target l1 (keeping order)
lt=Ludff; l1=L1udff; n=Nff; kguess=3; // target l1 (new order) MODIFIED KGUESS!
Call Compk;
Kudff=kguess;
kudff=Kudff^(1/(Pudff-1));
l1udff=L1udff*(1-kudff)/(1-kudff^(Pudff-1)); // first bundle size
Nffc=1+Log(1-Ludff*(1-Kudff)/L1udff)/Log(Kudff); // Point count (rounded up) check!
L1udffc=Ludff*(Kudff-1)/(Kudff^(Nff-1)-1); // recalc L1 (= target l1?) check!
l1udffc=L1udffc*(1-kudff)/(1-kudff^(Pudff-1)); // recalc l1 (= target l1?) check!
LNudff=L1udff*Kudff^(Nff-2); // Last interval size
lnudff=LNudff*(1-kudff)/(1-kudff^(Pudff-1))*kudff^(Pudff-2) ; // last virtual interval size
Printf('upper downstream wake farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Ludff,l1udff,lnudff,nff,kudff);
Printf('upper downstream wake farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Ludff,L1udff,LNudff,Nff,Kudff);
Printf('upper downstream wake farfield checks: l1=%g; L1=%g;N=%g',l1udffc,L1udffc,Nffc);

//* lower downstream wake farfield
//Kldff=Klff^0.7;//1;//Kff;//
Pldff=Pff;
Lldff=Sqrt((xldff-xldmf)^2+(yldff-yldmf)^2); // total length
L1ldff=LNldmf*(Pldff-1)/(Pldmf-1); // target L1 (keeping order)
Printf('culo %g %g',Pldff,Pldmf);
lt=Lldff; l1=L1ldff; n=Nff; kguess=3; // target l1 (new order) MODIFIED KGUESS!
Call Compk;
Kldff=kguess;
kldff=Kldff^(1/(Pldff-1));
l1ldff=L1ldff*(1-kldff)/(1-kldff^(Pldff-1)); // first bundle size
nffc=1+Log(1-Lldff*(1-Kldff)/L1ldff)/Log(Kldff); // Point count (rounded up) check!
L1ldffc=Lldff*(Kldff-1)/(Kldff^(nff-1)-1); // recalc L1 (= target l1?) check!
l1ldffc=L1ldffc*(1-kldff)/(1-kldff^(Pldff-1)); // recalc l1 (= target l1?) check!
LNldff=L1ldff*Kldff^(nff-2); // Last interval size
lnldff=LNldff*(1-kldff)/(1-kldff^(Pldff-1))*kldff^(Pldff-2) ; // last virtual interval size
Printf('lower downstream wake farfield: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Lldff,l1ldff,lnldff,nff,kldff);
Printf('lower downstream wake farfield: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Lldff,L1ldff,LNldff,nff,Kldff);
Printf('lower downstream wake farfield checks: l1=%g; L1=%g;N=%g',l1ldffc,L1ldffc,nffc);

//* * * DOWNSTREAM FRINGE REGION

If (fr>0)
  //* downstream wake
  //l1w=l1l*5; // would be nice to use last bump cell on airfoil
  L1i=LNw; // first bundle size
  Li=fr;
  Ki=ki^(Pii-1); // bundle progression ratio
  If (Ki==1)
    Ni=Floor(Li/L1i);
    L1i=Li/(Ni-1);
    l1i=L1i/(Pii-1);
    LNi=L1i; // Last bundle size
    lni=l1i; // last virtual interval size
    Else
    Ni=Ceil(1+Log(1-Li*(1-Ki)/L1i)/Log(Ki)); // Point count (rounded up)
    L1i=Li*(Ki-1)/(Ki^(Ni-1)-1); // recalc L1
    l1i=L1i*(1-ki)/(1-ki^(Pii-1)); // recalc l1
    LNi=L1i*Ki^(Ni-2); // Last bundle size
    lni=LNi*(1-ki)/(1-ki^(Pii-1))*ki^(Pii-2) ; // last virtual interval size
  EndIf
  ni=(Ni-1)*(Pii-1)+1;
  Printf('fringe region: L= %g; l1=%g; ln=%g; n=%g; kf=%g',Li,l1i,lni,ni,ki);
  Printf('fringe region: L= %g; L1=%g; LN=%g; N=%g; Kf=%g',Li,L1i,LNi,Ni,Ki);
EndIf

//* * * LINE DISCRETISATION (TRANSFINITE)

If (ibuas==1)
  Transfinite Line {uasL} = Nua Using Progression kbuas;
  ElseIf (ibuas==2)
  Transfinite Line {uasL} = Nua Using Bump kbuas;
  Else
  Transfinite Line {uasL} = Nua;
EndIf
If (ibuanf==1)
  //Transfinite Line {-uanfL} = Nua Using Progression kbuanf;
  //Transfinite Curve {-uanfL2,-uanfL1} = Nua Using Progression kbuanf;
  //Transfinite Curve {-uanfL2} = Nua-3 Using Progression kbuanf;
  //Transfinite Curve {-uanfL1} = 4 Using Progression kbuanf;
  Transfinite Curve {-uanfL} = Nua Using Progression kbuanf;
  ElseIf (ibuanf==2)
  Transfinite Line {uanfL} = Nua Using Bump kbuanf;
  //Transfinite Curve {uanfL1,uanfL2} = Nua Using Bump kbuanf;
  //Transfinite Curve {uanfL} = Nua Using Bump kbuanf;
  Else
  Transfinite Line {uanfL} = Nua;
  //Transfinite Curve {uanfL1,uanfL2} = Nua;
  //Transfinite Curve {uanfL1} = Nua-3;
  //Transfinite Curve {uanfL2} = 4;
  //Transfinite Curve {uanf} = Nua;
EndIf  
If (ibuamf==1)
  Transfinite Line {-uamfL} = Nua Using Progression kbuamf;
  //Transfinite Curve {-uamfL2,-uamfL1} = Nua Using Progression kbuamf;
  ElseIf (ibuamf==2)
  Transfinite Line {uamfL} = Nua Using Bump kbuamf;
  //Transfinite Curve {uamfL1,uamfL2} = Nua Using Bump kbuamf;
  Else
  Transfinite Line {uamfL} = Nua;
  //Transfinite Curve {uamfL,uamfL} = Nua;
EndIf  
Transfinite Line {uaffL} = Nua;
If (ibufs==1)
  Transfinite Line {-ufsL} = Nuf Using Progression kbufs;
  ElseIf (ibufs==2)
  Transfinite Line {ufsL} = Nuf Using Bump kbufs;
  Else
  Transfinite Line {ufsL} = Nuf;
EndIf
If (ibufnf==1)
  Transfinite Line {ufnfL} = Nuf Using Progression kbufnf;
  ElseIf (ibufnf==2)
  Transfinite Line {ufnfL} = Nuf Using Bump kbufnf;
  Else
  Transfinite Line {ufnfL} = Nuf;
EndIf  
If (ibufmf==1)
  Transfinite Line {ufmfL} = Nuf Using Progression kbufmf;
  ElseIf (ibufmf==2)
  Transfinite Line {ufmfL} = Nuf Using Bump kbufmf;
  Else
  Transfinite Line {ufmfL} = Nuf;
EndIf  
Transfinite Line {ufffL} = Nuf;

If (iblas==1)
  Transfinite Line {lasL} = Nla Using Progression kblas;
  ElseIf (iblas==2)
  Transfinite Line {lasL} = Nla Using Bump kblas;
  Else
  Transfinite Line {lasL} = Nla;
EndIf
If (iblanf==1)
  Transfinite Line {lanfL} = Nla Using Progression kblanf;
  //Transfinite Curve {lanfL1,lanfL2} = Nla Using Progression kblanf;
  ElseIf (iblanf==2)
  Transfinite Line {lanfL} = Nla Using Bump kblanf;
  //Transfinite Curve {lanfL1,lanfL2} = Nla Using Bump kblanf;
  Else
  Transfinite Line {lanfL} = Nla;
  //Transfinite Curve {lanfL1,lanfL2} = Nla;
EndIf  
If (iblamf==1)
  Transfinite Line {lamfL} = Nla Using Progression kblamf;
  //Transfinite Curve {lamfL1,lamfL2} = Nla Using Progression kblamf;
  ElseIf (iblamf==2)
  Transfinite Line {lamfL} = Nla Using Bump kblamf;
  //Transfinite Curve {lamfL1,lamfL2} = Nla Using Bump kblamf;
  Else
  Transfinite Line {lamfL} = Nla;
  //Transfinite Curve {lamfL1,lamfL2} = Nla;
EndIf  
Transfinite Line {laffL} = Nla;
If (iblfs==1)
  Transfinite Line {-lfsL} = Nlf Using Progression kblfs;
  ElseIf (iblfs==2)
  Transfinite Line {lfsL} = Nlf Using Bump kblfs;
  Else
  Transfinite Line {lfsL} = Nlf;
EndIf
If (iblfnf==1)
  Transfinite Line {lfnfL} = Nlf Using Progression kblfnf;
  ElseIf (iblfnf==2)
  Transfinite Line {lfnfL} = Nlf Using Bump kblfnf;
  Else
  Transfinite Line {lfnfL} = Nlf;
EndIf  
If (iblfmf==1)
  Transfinite Line {lfmfL} = Nlf Using Progression kblfmf;
  ElseIf (iblfmf==2)
  Transfinite Line {lfmfL} = Nlf Using Bump kblfmf;
  Else
  Transfinite Line {lfmfL} = Nlf;
EndIf  
Transfinite Line {lfffL} = Nlf;
//Transfinite Line {-PLf} = Nal Using Progression KfPLf;
//
Transfinite Line {lenfL} = Nnf Using Progression Klenf;
Transfinite Line {utenfL} = Nnf Using Progression Kutenf;
Transfinite Line {ujnfL} = Nnf Using Progression Kujnf;
Transfinite Line {ltenfL} = Nnf Using Progression Kltenf;
Transfinite Line {ljnfL} = Nnf Using Progression Kljnf;
Transfinite Line {udnfL} = Nnf Using Progression Kudnf;
Transfinite Line {ldnfL} = Nnf Using Progression Kldnf;

Transfinite Line {leffL} = Nff Using Progression Kleff;
Transfinite Line {ujffL} = Nff Using Progression Kujff;
Transfinite Line {uteffL} = Nff Using Progression Kuteff;
Transfinite Line {ljffL} = Nff Using Progression Kljff;
Transfinite Line {lteffL} = Nff Using Progression Klteff;
Transfinite Line {udffL} = Nff Using Progression Kudff;//Kuff^0.5;//Kuff;//Kuteff;
Transfinite Line {ldffL} = Nff Using Progression Kldff;//Klff^0.5;//Klff;//Klaff;
//

Transfinite Line {lemfL} = Nmf Using Progression Klemf;
Transfinite Line {ujmfL} = Nmf Using Progression Kujmf;
Transfinite Line {utemfL} = Nmf Using Progression Kutemf;
Transfinite Line {ljmfL} = Nmf Using Progression Kljmf;
Transfinite Line {ltemfL} = Nmf Using Progression Kltemf;
Transfinite Line {udmfL} = Nmf Using Progression 1;//Kudmf;
Transfinite Line {ldmfL} = Nmf Using Progression 1;//Kldmf;
//

Transfinite Line {wL} = Nw Using Progression Kw;
Transfinite Line {uwnfL} = Nw Using Progression Kw;//Kw^0.95;
Transfinite Line {lwnfL} = Nw Using Progression Kw;//Kw^0.95;
Transfinite Line {uwmfL} = Nw Using Progression Kw;//Kw^0.25;//KfPUa;//
Transfinite Line {lwmfL} = Nw Using Progression Kw;//Kw^0.25;//KfPLa;//
Transfinite Line {uwffL} = Nw Using Progression 1;//Kw^0.25;//KfPUa;//
Transfinite Line {lwffL} = Nw Using Progression 1;//Kw^0.25;//KfPLa;//
//

//* FRINGE REGION
If (fr>0)
  Transfinite Line {uiffL} = Nff Using Progression Kudff;
  Transfinite Line {liffL} = Nff Using Progression Kldff;
  Transfinite Line {uimfL} = Nmf Using Progression Kudmf;
  Transfinite Line {limfL} = Nmf Using Progression Kldmf;
  Transfinite Line {uinfL} = Nnf Using Progression Kudnf;
  Transfinite Line {linfL} = Nnf Using Progression Kldnf;
  Transfinite Line {wiL} = Ni Using Progression Ki;
  Transfinite Line {uwinfL} = Ni Using Progression Ki;
  Transfinite Line {lwinfL} = Ni Using Progression Ki;
  Transfinite Line {uwimfL} = Ni Using Progression Ki;
  Transfinite Line {lwimfL} = Ni Using Progression Ki;
  Transfinite Line {uwiffL} = Ni Using Progression Ki;
  Transfinite Line {lwiffL} = Ni Using Progression Ki;
EndIf

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* LOOPS
uwffLl = newll; Line Loop(uwffLl) = {uteffL, uwffL, -udffL, -uwmfL};
uaffLl = newll; Line Loop(uaffLl) = {ujffL, uaffL, -uteffL, -uamfL};
//uaffLl = newll; Curve Loop(uaffLl) = {ujffL, uaffL, -uteffL, -uamfL2, -uamfL1};
ufffLl = newll; Line Loop(ufffLl) = {leffL, ufffL, -ujffL, -ufmfL};
lwffLl = newll; Line Loop(lwffLl) = {-lteffL, lwmfL, ldffL, -lwffL};
laffLl = newll; Line Loop(laffLl) = {-ljffL, lamfL, lteffL, -laffL};
//laffLl = newll; Curve Loop(laffLl) = {-ljffL, lamfL1, lamfL2, lteffL, -laffL};
lfffLl = newll; Line Loop(lfffLl) = {-leffL, lfmfL, ljffL, -lfffL};
//
uwmfLl = newll; Line Loop(uwmfLl) = {utemfL, uwmfL, -udmfL, -uwnfL};
uamfLl = newll; Line Loop(uamfLl) = {ujmfL, uamfL, -utemfL, -uanfL};
//uamfLl = newll; Curve Loop(uamfLl) = {ujmfL, uamfL, -utemfL, -uanfL2, -uanfL1};
//uamfLl = newll; Curve Loop(uamfLl) = {ujmfL, uamfL1, uamfL2, -utemfL, -uanfL2, -uanfL1};
ufmfLl = newll; Line Loop(ufmfLl) = {lemfL, ufmfL, -ujmfL, -ufnfL};
lwmfLl = newll; Line Loop(lwmfLl) = {-ltemfL, lwnfL, ldmfL, -lwmfL};
lamfLl = newll; Line Loop(lamfLl) = {-ljmfL, lanfL, ltemfL, -lamfL};
//lamfLl = newll; Curve Loop(lamfLl) = {-ljmfL, lanfL1, lanfL2, ltemfL, -lamfL2, -lamfL1};
lfmfLl = newll; Line Loop(lfmfLl) = {-lemfL, lfnfL, ljmfL, -lfmfL};
//
uwnfLl = newll; Line Loop(uwnfLl) = {utenfL, uwnfL, -udnfL, -wL};
uanfLl = newll; Line Loop(uanfLl) = {ujnfL, uanfL, -utenfL, uasL};
//uanfLl = newll; Curve Loop(uanfLl) = {ujnfL, uanfL1, uanfL2, -utenfL, uasL};
ufnfLl = newll; Line Loop(ufnfLl) = {lenfL, ufnfL, -ujnfL, ufsL};
lwnfLl = newll; Line Loop(lwnfLl) = {-ltenfL, wL, ldnfL, -lwnfL};
lanfLl = newll; Line Loop(lanfLl) = {-ljnfL, -lasL, ltenfL, -lanfL};
//lanfLl = newll; Curve Loop(lanfLl) = {-ljnfL, -lasL, ltenfL, -lanfL2, -lanfL1};
lfnfLl = newll; Line Loop(lfnfLl) = {-lenfL, -lfsL, ljnfL, -lfnfL};
//Printf('%g %g %g %g %g',lfmfLl,-lemfL, lfnfL, ljmfL, -lfmfL);

If (fr>0)
  uiffLl = newll; Line Loop(uiffLl) = {udffL, uwiffL, -uiffL, -uwimfL};
  uimfLl = newll; Line Loop(uimfLl) = {udmfL, uwimfL, -uimfL, -uwinfL};
  uinfLl = newll; Line Loop(uinfLl) = {udnfL, uwinfL, -uinfL, -wiL};
  liffLl = newll; Line Loop(liffLl) = {-ldffL, lwimfL, liffL, -lwiffL};
  limfLl = newll; Line Loop(limfLl) = {-ldmfL, lwinfL, limfL, -lwimfL};
  linfLl = newll; Line Loop(linfLl) = {-ldnfL, wiL, linfL, -lwinfL};
EndIf

////Printf('%g',LlWl);
////
////Color Red{Line{LWLine};}

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\\
//* SURFACES
uwffS = news; Plane Surface(uwffS) = {uwffLl};
uaffS = news; Plane Surface(uaffS) = {uaffLl};
ufffS = news; Plane Surface(ufffS) = {ufffLl};
lwffS = news; Plane Surface(lwffS) = {lwffLl};
laffS = news; Plane Surface(laffS) = {laffLl};
lfffS = news; Plane Surface(lfffS) = {lfffLl};
//
uwmfS = news; Plane Surface(uwmfS) = {uwmfLl};
uamfS = news; Plane Surface(uamfS) = {uamfLl};
ufmfS = news; Plane Surface(ufmfS) = {ufmfLl};
lwmfS = news; Plane Surface(lwmfS) = {lwmfLl};
lamfS = news; Plane Surface(lamfS) = {lamfLl};
lfmfS = news; Plane Surface(lfmfS) = {lfmfLl};
//
uwnfS = news; Plane Surface(uwnfS) = {uwnfLl};
uanfS = news; Plane Surface(uanfS) = {uanfLl};
//uanfS = news; Compound Surface(uanfS) = {uanfLl};
ufnfS = news; Plane Surface(ufnfS) = {ufnfLl};
lwnfS = news; Plane Surface(lwnfS) = {lwnfLl};
lanfS = news; Plane Surface(lanfS) = {lanfLl};
lfnfS = news; Plane Surface(lfnfS) = {lfnfLl};

If (fr>0)
  uiffS = news; Plane Surface(uiffS) = {uiffLl};
  uimfS = news; Plane Surface(uimfS) = {uimfLl};
  uinfS = news; Plane Surface(uinfS) = {uinfLl};
  liffS = news; Plane Surface(liffS) = {liffLl};
  limfS = news; Plane Surface(limfS) = {limfLl};
  linfS = news; Plane Surface(linfS) = {linfLl};
EndIf
//
//Hide {Surface {SFFuf};}
//Recursive Hide {Line {usPLine,lsPLine};}
//
Transfinite Surface {uwffS,uaffS,ufffS,lwffS,laffS,lfffS};
Recombine Surface {uwffS,uaffS,ufffS,lwffS,laffS,lfffS};
Transfinite Surface {uwmfS,ufmfS,uamfS,lwmfS,lamfS,lfmfS};
Recombine Surface {uwmfS,uamfS,ufmfS,lwmfS,lamfS,lfmfS};
Transfinite Surface {uwnfS,ufnfS,uanfS,lwnfS,lanfS,lfnfS};
Recombine Surface {uwnfS,uanfS,ufnfS,lwnfS,lanfS,lfnfS};
If (fr>0)
  Transfinite Surface {uinfS,uimfS,uiffS,linfS,limfS,liffS};
  Recombine Surface {uinfS,uimfS,uiffS,linfS,limfS,liffS};
EndIf

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\\
//* PHYSICAL ENTITIES

If (fr>0)

  //Color Red{Physical Line("Inlet") = {ufffL,uaffL,lfffL,laffL};}
  //Color Blue{Physical Line("Outlet") = {uiffL,uimfL,uinfL,liffL,limfL,linfL};}
  //Physical Line("AirfoilUS") = {uasL,ufsL};
  //Physical Line("AirfoilLS") = {lasL,lfsL};
  //Physical Line("UpperBoundary") = {uwiffL,uwffL};
  //Physical Line("LowerBoundary") = {lwiffL,lwffL};

  //Physical Surface("FarField") = {uiffS,uwffS,uaffS,ufffS,liffS,lwffS,laffS,lfffS};
  //Physical Surface("UpperFarFieldWake") = {uwffS};
  //Physical Surface("UpperFarFieldAft") = {uaffS};
  //Physical Surface("UpperFarFieldFor") = {ufffS};
  //Physical Surface("LowerFarFieldWake") = {lwffS};
  //Physical Surface("LowerFarFieldAft") = {laffS};
  //Physical Surface("LowerFarFieldFor") = {lfffS};

  //Physical Surface("MidField") = {uwmfS,uamfS,ufmfS,lwmfS,lamfS,lfmfS};
  //Physical Surface("UpperMidFieldWake") = {uwmfS};
  //Physical Surface("MidFieldExceptUpperWake") = {uimfS,uamfS,ufmfS,limfS,lwmfS,lamfS,lfmfS};
  //Physical Surface("UpperMidFieldAft") = {uamfS};
  //Physical Surface("UpperMidFieldFor") = {ufmfS};
  //Physical Surface("LowerMidFieldWake") = {lwmfS};
  //Physical Surface("LowerMidFieldAft") = {lamfS};
  //Physical Surface("LowerMidFieldFor") = {lfmfS};

  //Physical Surface("NearField") = {uwnfS,uanfS,ufnfS,lwnfS,lanfS,lfnfS};
  //Physical Surface("UpperNearField") = {uanfS,ufnfS};
  //Physical Surface("UpperNearFieldAft") = {uanfS};
  //Physical Surface("UpperNearFieldFor") = {ufnfS};
  //Physical Surface("LowerNearField") = {lanfS,lfnfS};
  //Physical Surface("LowerNearFieldAft") = {lanfS};
  //Physical Surface("LowerNearFieldFor") = {lfnfS};

  //Physical Surface("Wake") = {uinfS,uwnfS,linfS,lwnfS};
  //Physical Surface("UpperWake") = {uwnfS};
  //Physical Surface("LowerWake") = {lwnfS};

Else

  //Color Red{Physical Line("Inlet") = {ufffL,uaffL,lfffL,laffL};}
  //Color Blue{Physical Line("Outlet") = {udffL,udmfL,udnfL,ldffL,ldmfL,ldnfL};}
  //Physical Line("AirfoilUS") = {uasL,ufsL};
  //Physical Line("AirfoilLS") = {lasL,lfsL};
  //Physical Line("UpperBoundary") = {uwffL};
  //Physical Line("LowerBoundary") = {lwffL};

  //Physical Surface("FluidDomain") = {};

  //Physical Surface("FarField") = {uwffS,uaffS,ufffS,lwffS,laffS,lfffS};
  //Physical Surface("UpperFarFieldWake") = {uwffS};
  //Physical Surface("UpperFarFieldAft") = {uaffS};
  //Physical Surface("UpperFarFieldFor") = {ufffS};
  //Physical Surface("LowerFarFieldWake") = {lwffS};
  //Physical Surface("LowerFarFieldAft") = {laffS};
  //Physical Surface("LowerFarFieldFor") = {lfffS};

  //Physical Surface("MidField") = {uwmfS,uamfS,ufmfS,lwmfS,lamfS,lfmfS};
  //Physical Surface("UpperMidFieldWake") = {uwmfS};
  //Physical Surface("MidFieldExceptUpperWake") = {uamfS,ufmfS,lwmfS,lamfS,lfmfS};
  //Physical Surface("UpperMidFieldAft") = {uamfS};
  //Physical Surface("UpperMidFieldFor") = {ufmfS};
  //Physical Surface("LowerMidFieldWake") = {lwmfS};
  //Physical Surface("LowerMidFieldAft") = {lamfS};
  //Physical Surface("LowerMidFieldFor") = {lfmfS};

  //Physical Surface("NearField") = {uwnfS,uanfS,ufnfS,lwnfS,lanfS,lfnfS};
  //Physical Surface("UpperNearField") = {uanfS,ufnfS};
  //Physical Surface("UpperNearFieldAft") = {uanfS};
  //Physical Surface("UpperNearFieldFor") = {ufnfS};
  //Physical Surface("LowerNearField") = {lanfS,lfnfS};
  //Physical Surface("LowerNearFieldAft") = {lanfS};
  //Physical Surface("LowerNearFieldFor") = {lfnfS};
  //Physical Surface("Wake") = {uwnfS,lwnfS};
  //Physical Surface("UpperWake") = {uwnfS};
  //Physical Surface("LowerWake") = {lwnfS};

EndIf

If (fr>0)
  Extrude {0,0,1.0}
  {
    Surface{uiffS,uwffS,uaffS,ufffS,liffS,lwffS,laffS,lfffS,uwmfS,uimfS,uamfS,ufmfS,limfS,lwmfS,lamfS,lfmfS,uanfS,ufnfS,lanfS,lfnfS,uinfS,uwnfS,linfS,lwnfS};
    Layers{1};
    Recombine;
  }
  Physical Surface("in") = {162, 184, 280, 258};
  Physical Surface("out") = {122, 320, 562, 606, 386, 210};
  Physical Surface("top") = {140, 118};
  Physical Surface("bottom") = {236, 214};
  Physical Surface("uppsurf") = {478, 500};
  Physical Surface("lowsurf") = {514, 536};
  Physical Surface("front") = {uiffS,uwffS,uaffS,ufffS,liffS,lwffS,laffS,lfffS,uwmfS,uimfS,uamfS,ufmfS,limfS,lwmfS,lamfS,lfmfS,uanfS,ufnfS,lanfS,lfnfS,uinfS,uwnfS,linfS,lwnfS};
  Physical Surface("back") = {171, 193, 281, 259, 347, 369, 457, 435, 479, 501, 545, 523, 149, 303, 589, 633, 413, 237, 127, 325, 567, 611, 391, 215};
  Physical Volume("fluid") = {1:24};
Else
  Extrude {0,0,1.0}
  {
    Surface{uwffS,uaffS,ufffS,lwffS,laffS,lfffS,uwmfS,uamfS,ufmfS,lwmfS,lamfS,lfmfS,uanfS,ufnfS,lanfS,lfnfS,uwnfS,lwnfS};
    Layers{1};
    Recombine;
  }
EndIf

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
//* * * GENERATE MESH
//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *//
Mesh.HighOrderOptimize=2;
Mesh.NumSubEdges=nord;
Mesh 1;
SetOrder nord;
Mesh 2;
SetOrder nord;
Mesh 3;
OptimizeMesh "Gmsh"; // OptimizeMesh "NetGen";
Printf('%g',Mesh.NumSubEdges);
//RelocateMesh "*";
Printf('N upper surface for= %g',Nuf);
Printf('N upper surface aft= %g',Nua);
Printf('N lower surface for= %g',Nlf);
Printf('N lower surface aft= %g',Nla);
Printf('N nearfield= %g',Nnf);
Printf('N midfield= %g',Nmf);
Printf('N farfield= %g',Nff);
Printf('N wake= %g',Nw);
If (fr>0)
  Printf('N fringe= %g',Ni);
  Else
  Printf('No fringe region');
EndIf
