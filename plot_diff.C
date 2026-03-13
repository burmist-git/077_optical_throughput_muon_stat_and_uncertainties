void read_data(TString fname, TGraphErrors *gr);

Int_t plot_diff(){

  TGraphErrors *gr = new TGraphErrors();
  TGraphErrors *gr20p = new TGraphErrors();
  TGraphErrors *gr50p = new TGraphErrors();
  TGraphErrors *grcut = new TGraphErrors();
  
  read_data("./data/muon-_20deg_0deg_run000002___cta-prod6-2156m-LaPalma-lst-dark.h5.csv", gr);
  read_data("./data/muon-_20deg_0deg_run000002___cta-prod6-2156m-LaPalma-lst-dark-align-deg-20p.h5.csv", gr20p);
  read_data("./data/muon-_20deg_0deg_run000002___cta-prod6-2156m-LaPalma-lst-dark-align-deg-50p.h5.csv", gr50p);
  read_data("./data/muon-_0deg_0deg_run000003___cta-prod6-2147m-Paranal-lst-dark-ref-degraded-0.8.h5.csv", grcut);
   
  TCanvas *c1;

  gr->SetMarkerColor(kBlack);
  gr20p->SetMarkerColor(kBlue+2);
  gr50p->SetMarkerColor(kRed+2);

  gr->SetLineColor(kBlack);
  gr20p->SetLineColor(kBlue+2);
  gr50p->SetLineColor(kRed+2);
  
  gr->SetLineWidth(2.0);
  gr20p->SetLineWidth(2.0);
  gr50p->SetLineWidth(2.0);
  

  grcut->SetLineColor(kMagenta+2);
  grcut->SetLineWidth(2.0);
  
  TMultiGraph *mg = new TMultiGraph();
  mg->Add(gr);
  //mg->Add(grcut);
  mg->Add(gr20p);
  mg->Add(gr50p);
  mg->Draw("AP");
  mg->GetXaxis()->SetTitle("Ring radius, deg");
  mg->GetYaxis()->SetTitle("Ring width, deg");

  TLegend *leg = new TLegend(0.6,0.6,0.9,0.9,"","brNDC");
  leg->AddEntry(gr, "Nominal (0.0046)", "apl");
  //leg->AddEntry(grcut, "Nominal (0.0046) on-axis muons", "apl");
  leg->AddEntry(gr20p, "Degraded by 20% (0.0055)", "apl");
  leg->AddEntry(gr50p, "Degraded by 50% (0.0069)", "apl");
  leg->Draw();  

  
  //mg->Add(grcut);
  //071_chord_fit/plots_diff.C:  h1_reco_y_m_true->SetMarkerColor(kGreen+2);
  //069_Optimum_circular_fit_Chaudhuri_Taubin/fit_muon_ring.C:  gr_app_r0->SetMarkerSize(2.0);
  //069_Optimum_circular_fit_Chaudhuri_Taubin/fit_muon_ring.C:  gr_app_average_r0->SetMarkerStyle(43);

  return 0;
}

void read_data(TString fname, TGraphErrors *gr){
  string mot;
  ifstream fFile(fname);
  Double_t x, y;
  Double_t xerr, yerr;
  Int_t point_counter = 0;
  if(fFile.is_open()){
    fFile>>mot>>mot>>mot>>mot;
    while(fFile>>x>>y>>xerr>>yerr){
      gr->SetPoint(point_counter,x,y);
      gr->SetPointError(point_counter,
                        xerr,
                        yerr);
      point_counter++;
    }
    fFile.close();
  }
}
