//x y xe ye
//0.8051020408163265 0.06162447695507728 0.8051020408163265 0.06162447695507728
//0.8153061224489797 0.1208299510424685 0.8153061224489797 0.1208299510424685
// . . .

TGraphErrors *_gr = new TGraphErrors();
TGraph *_gr_fit = new TGraph();

Double_t _angular_cut_off_max = 1.17;
Double_t _angular_cut_off_min = 0.9;

void fit_ring_with_Minuit(Double_t x0in, Double_t y0in, Double_t Rin,
			  Double_t &x0out, Double_t &y0out, Double_t &Rout,
			  Double_t &x0outerr, Double_t &y0outerr, Double_t &Routerr);

void fcn(int &npar, double *gin, double &f, double *par, int iflag);
double function_to_fit(double x, double *par);
void read_data(TString fname);
void get_fit_data(Double_t xmin, Double_t xmax, Int_t npoints,
		  Double_t A,  Double_t theta_critical,  Double_t sigma_inf);

Int_t fit(){
  //read_data("./data/muon-_0deg_0deg_run000003___cta-prod6-2147m-Paranal-lst-dark-ref-degraded-0.8.h5.csv");
  //read_data("./data/muon-_0deg_0deg_run000002___cta-prod6-2147m-Paranal-mst-fc-dark-ref-degraded-0.8.h5.csv");
  //read_data("./data/muon-_0deg_0deg_run000001___cta-prod6-2147m-Paranal-mst-nc-dark-ref-degraded-0.8.h5.csv");

  //read_data("./data/muon-_20deg_0deg_run000002___cta-prod6-2156m-LaPalma-lst-dark.h5.csv");
  read_data("./data/muon-_20deg_0deg_run000002___cta-prod6-2156m-LaPalma-lst-dark-align-deg-20p.h5.csv");
  //read_data("./data/muon-_20deg_0deg_run000002___cta-prod6-2156m-LaPalma-lst-dark-align-deg-50p.h5.csv");
  
  Double_t Ain = 0.0684895;
  Double_t theta_critical_in = 1.15;
  Double_t sigma_inf_in = 0.0247842;
  
  Double_t Aout, theta_critical_out, sigma_inf_out;
  Double_t Aouterr, theta_critical_outerr, sigma_inf_outerr;
  

  fit_ring_with_Minuit( Ain,  theta_critical_in,  sigma_inf_in,
		        Aout, theta_critical_out, sigma_inf_out,
		        Aouterr, theta_critical_outerr, sigma_inf_outerr);

  get_fit_data( 0.8, 1.3, 1000,
	        Aout,  theta_critical_out,  sigma_inf_out);

  
  cout<<"Aout               = "<<Aout<<endl
      <<"theta_critical_out = "<<theta_critical_out<<endl
      <<"sigma_inf_out      = "<<sigma_inf_out<<endl;
  
  TCanvas *c1;

  TMultiGraph *mg = new TMultiGraph();
  mg->Add(_gr);
  mg->Add(_gr_fit);
  mg->Draw("AP");
  mg->GetXaxis()->SetTitle("Ring radius, deg");
  mg->GetYaxis()->SetTitle("Ring width, deg");
  
  return 0;
}

void read_data(TString fname){
  string mot;
  ifstream fFile(fname);
  Double_t x, y;
  Double_t xerr, yerr;
  Int_t point_counter = 0;
  if(fFile.is_open()){
    fFile>>mot>>mot>>mot>>mot;
    while(fFile>>x>>y>>xerr>>yerr){
      _gr->SetPoint(point_counter,x,y/x);
      _gr->SetPointError(point_counter,
                        xerr,
                        yerr);
      point_counter++;
    }
    fFile.close();
  }
}

void get_fit_data(Double_t xmin, Double_t xmax, Int_t npoints,
		  Double_t A,  Double_t theta_critical,  Double_t sigma_inf){
  Double_t x, y;
  double par[3]={A, theta_critical, sigma_inf};
  for(Int_t i = 0; i<npoints; i++){
    x = xmin + (xmax - xmin)/(npoints - 1)*i;
    y = function_to_fit( x, par);
    _gr_fit->SetPoint(i,x,y);
  }
}


void fcn(int &npar, double *gin, double &f, double *par, int iflag){
  double chisq = 0;
  double x, y, xe, ye, etot;
  double delta;
  for (int i = 0; i<_gr->GetN(); i++){
    _gr->GetPoint(i, x, y);
    xe=_gr->GetErrorX(i);
    ye=_gr->GetErrorY(i);
    etot = TMath::Sqrt(xe*xe + ye*ye);
    if( x<=_angular_cut_off_max && x >=_angular_cut_off_min){
      delta = (function_to_fit(x, par) - y) / etot / etot;
      delta *= delta;
      chisq += delta;
    }
  }  
  f = chisq;
}

double function_to_fit(double x, double *par){
  double A = par[0];
  double theta_critical = par[1];
  double sigma_inf = par[2];
  if(x<0)
    return 0.0;
  if(x>theta_critical)
    return 0.0;
  //
  return A*TMath::Sqrt(1 - x * x / theta_critical / theta_critical) / x + sigma_inf;
}

void fit_ring_with_Minuit(Double_t Ain, Double_t theta_critical_in, Double_t sigma_inf_in,
			  Double_t &Aout, Double_t &theta_critical_out, Double_t &sigma_inf_out,
			  Double_t &Aouterr, Double_t &theta_critical_outerr, Double_t &sigma_inf_outerr){
  //
  Int_t npar = 3;
  TMinuit *gMinuit = new TMinuit(npar);
  gMinuit->SetPrintLevel(-1.0);
  gMinuit->SetFCN(fcn); 
  double arglist[10];
  int ierflg = 0;
  arglist[0] = 1;
  gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);
  // 
  // Set starting values and step sizes for parameters
  gMinuit->mnparm(0, "A", Ain, 0.01, 0,0,ierflg);
  gMinuit->mnparm(1, "theta_critical", theta_critical_in, 0.01, 0,0,ierflg);
  gMinuit->mnparm(2, "sigma_inf", sigma_inf_in, 0.01, 0,0,ierflg);
  //

  // Now ready for minimization step
  arglist[0] = 50000;
  arglist[1] = 1.;
  gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);
  
  // Print results
  double amin,edm,errdef;
  int nvpar,nparx,icstat;
  gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
  //gMinuit->mnprin(3,amin);
  //
  gMinuit->GetParameter(0, Aout, Aouterr);
  gMinuit->GetParameter(1, theta_critical_out, theta_critical_outerr);
  gMinuit->GetParameter(2, sigma_inf_out, sigma_inf_outerr);
  //
}
