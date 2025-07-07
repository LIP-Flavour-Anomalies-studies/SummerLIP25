#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "RooFit.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooAddPdf.h"
#include "RooArgList.h"
#include "RooArgSet.h"
#include "TLatex.h"
#include "RooChi2Var.h"
#include "RooAbsData.h"

using namespace std;
using namespace RooFit; 

void bMass_mc(){
    // --- Open Files and Get Trees ---
	TFile *f_mc = new TFile("/lstore/cms/boletti/Run3-ntuples/reco_ntuple2_LMNR_1.root", "read");
	TTree *t_mc = (TTree*)f_mc->Get("ntuple");

    // --- Declare variables for simulation ---
    double mc_bMass, mc_bBarMass, mc_tagB0;
    double mc_bVtxCL;
    double mc_mumuMass;
    double mc_truthMatchMum, mc_truthMatchMup, mc_truthMatchTrkm, mc_truthMatchTrkp;

    t_mc->SetBranchAddress("bMass", &mc_bMass);
    t_mc->SetBranchAddress("bBarMass", &mc_bBarMass);
    t_mc->SetBranchAddress("tagB0", &mc_tagB0);
    t_mc->SetBranchAddress("bVtxCL", &mc_bVtxCL);
    t_mc->SetBranchAddress("mumuMass", &mc_mumuMass);

    t_mc->SetBranchAddress("truthMatchMum", &mc_truthMatchMum);
    t_mc->SetBranchAddress("truthMatchMup", &mc_truthMatchMup);
    t_mc->SetBranchAddress("truthMatchTrkm", &mc_truthMatchTrkm);
    t_mc->SetBranchAddress("truthMatchTrkp", &mc_truthMatchTrkp);

    double mmin = 5;
    double mmax = 5.6;

    TH1D *h_mc = new TH1D("h_mc", "B0 Mass (MC Truth Matched)", 100, mmin, mmax);

    // --- Fill MC histogram ---
    cout << "Processing MC Events..." << endl;
    Long64_t nEntries_mc = t_mc->GetEntries();
    int matchedMC = 0;
    for(Long64_t i = 0; i < nEntries_mc; i++){    
        t_mc->GetEntry(i);

        // Truth matching cuts first to isolate signal
        if (mc_truthMatchMum == 0) continue;
        if (mc_truthMatchMup == 0) continue;
        if (mc_truthMatchTrkm == 0) continue;
        if (mc_truthMatchTrkp == 0) continue;

        matchedMC++;

        //Selection cuts
        //if (mc_bVtxCL < 0.01) continue; // skip events with poor vertex fit   

        double bMass_mc = (mc_tagB0 == 1) ? mc_bMass : mc_bBarMass;
		if (bMass_mc < mmin || bMass_mc > mmax) continue;

        h_mc->Fill(bMass_mc);
    }
    cout << "Total MC events: " << nEntries_mc << endl;
    cout << "Fully truth-matched MC events: " << matchedMC << endl;


    // --- Perform the Fit ---
    RooRealVar mass("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}");
    RooArgList args(mass);
    RooDataHist dh("dh", "dh", args, h_mc);

    // backgorund model 
    RooRealVar lambda("lambda", "lambda", -0.3, -4.0, 0.0);
    RooExponential background("background", "background", mass, lambda);

    // signal model (Gaussian + Crystal ball)
    RooRealVar mean("mean", "mean", 0.5*(mmin+mmax), mmin, mmax);
    RooRealVar sigma("sigma", "sigma", 0.05*(mmax-mmin),0.,0.5*(mmax-mmin));
    RooRealVar sigma2("sigma2", "sigma2", 0.08*(mmax-mmin),0.,0.5*(mmax-mmin));
    RooRealVar alpha("alpha", "alpha", 1.0, 0.0, 5.); // use positive if tail on RHS
    RooRealVar n("n", "n", 5.0, 0.5, 20);

    RooGaussian gaussian("gaussian", "gaussian", mass, mean, sigma2);
    RooCrystalBall cb("cb", "cb", mass, mean, sigma, alpha, n);

    // fraction of Crystal ball
    RooRealVar frac("frac", "frac", 0.5, 0., 1.);
    RooAddPdf signal("signal", "signal", RooArgList(cb, gaussian), frac);

    // variables for number of signal and background events
    double totalEntries = dh.sumEntries();
    double n_signal_initial = 0.8 * totalEntries;
    double n_back_initial = 0.2 * totalEntries;
    RooRealVar n_signal("n_signal", "n_signal", n_signal_initial, 0., totalEntries);
    RooRealVar n_back("n_back", "n_back", n_back_initial, 0., totalEntries);

    // sum signal and background models 
    RooAddPdf model("model", "model", RooArgList(signal, background), RooArgList(n_signal, n_back));

    // fit 
    model.fitTo(dh);

    RooPlot* frame = mass.frame();
    frame->SetTitle("B^{0} mass");
    dh.plotOn(frame);
    model.plotOn(frame, Name("model"));
    model.plotOn(frame, Name("modelBkg"), Components("background"), LineStyle(kDashed), LineColor(kGreen));
    model.plotOn(frame, Name("modelSig"), Components("signal"), LineStyle(kDashed), LineColor(kRed));
    //model.paramOn(frame, Layout(0.6, 0.9, 0.9));  // Optional

    TCanvas *c_mc = new TCanvas("c_mc", "");
    frame->Draw();

    //Draw a caption
    TLegend *legend = new TLegend(0.65,0.6,0.88,0.85);
    legend->SetBorderSize(0);
    legend->SetTextFont(40);
    legend->SetTextSize(0.04);
    legend->AddEntry(frame->findObject("dh"),"Data","1pe");
    legend->AddEntry(frame->findObject("modelBkg"),"Background fit","1pe");
    legend->AddEntry(frame->findObject("modelSig"),"Signal fit","1pe");
    legend->AddEntry(frame->findObject("model"),"Global fit","1pe");
    legend->Draw();

    //Display info and fit results
    TLatex *L = new TLatex();
    L->SetNDC();
    L->SetTextSize(0.03);
    L->DrawLatex(0.15,0.8, Form("Y_{S}: %.0f #pm %.0f events", n_signal.getVal(), n_signal.getError()));
    L->DrawLatex(0.15,0.75, Form("Y_{B}: %.0f #pm %.0f events", n_back.getVal(),n_back.getError()));
    L->DrawLatex(0.15,0.70, Form("#lambda: %5.3f #pm %5.3f GeV^{-1}", lambda.getVal(),lambda.getError()));
    L->DrawLatex(0.15,0.65, Form("mass: %5.3f #pm %5.3f GeV/c^{2}", mean.getVal(),mean.getError()));
    L->DrawLatex(0.15,0.60, Form("#sigma_{g}: %5.3f #pm %5.3f MeV/c^{2}", sigma2.getVal()*1000,sigma2.getError()*1000));
    L->DrawLatex(0.15,0.55, Form("#sigma_{cb}: %5.3f #pm %5.3f MeV/c^{2}", sigma.getVal()*1000,sigma.getError()*1000));
    L->DrawLatex(0.15,0.50, Form("n: %5.3f #pm %5.3f", n.getVal(), n.getError()));
    L->DrawLatex(0.15,0.45, Form("#alpha: %5.3f #pm %5.3f", alpha.getVal(), alpha.getError()));
    RooChi2Var chi("chi","chi^2", model, dh);
    int variables = 5;  //This is the number of free parameters in our model, and the
                        // number of degrees of freedom is the the number of points in our
                        // model minus the free parameters
    L->DrawLatex(0.15,0.40,Form("#chi^{2}/ndf: %.2f", chi.getVal()/(100.-variables)));
    
    c_mc->Draw();
    c_mc->SaveAs("fit_mc.png"); 
    f_mc->Close();

}