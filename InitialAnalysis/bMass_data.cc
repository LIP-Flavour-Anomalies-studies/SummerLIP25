#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;

void bMass_data(){
    // --- Open Files and Get Trees ---
    TFile *f_data = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root", "read");
	TTree *t_data = (TTree*)f_data->Get("ntuple");

    // --- Declare variables for data ---
    double data_bMass, data_bBarMass, data_tagB0;
    double data_bVtxCL;
    double data_mumuMass;
    double data_bCosAlphaBS;
    double data_bLBS;

    
    t_data->SetBranchAddress("bMass", &data_bMass);
    t_data->SetBranchAddress("bBarMass", &data_bBarMass);
    t_data->SetBranchAddress("tagB0", &data_tagB0);
    t_data->SetBranchAddress("bVtxCL", &data_bVtxCL);
    t_data->SetBranchAddress("mumuMass", &data_mumuMass);
    t_data->SetBranchAddress("bCosAlphaBS", &data_bCosAlphaBS);
    t_data->SetBranchAddress("bLBS", &data_bLBS);

    // --- Create histogram ---
    TH1D *h = new TH1D("h", "", 100, 4.8, 5.8);
    h->GetXaxis()->SetTitle("m(B^{0}) Data [GeV/c^{2}]");
    h->GetYaxis()->SetTitle("Events");


    Long64_t nEntries_data = t_data->GetEntries();
    int events_selected = 0;
    for(Long64_t i = 0; i < nEntries_data; i++){    
        t_data->GetEntry(i);

        //Selection cuts
        if (data_bVtxCL < 0.1) continue; 
        if (data_bCosAlphaBS < 0.985) continue;
        if (data_bLBS < 0.04) continue;  

        double mass_b = (data_tagB0 == 1) ? data_bMass : data_bBarMass;
		if (mass_b < 5.0 || mass_b > 5.6) continue;

        events_selected++;

        h->Fill(mass_b);
    }
    cout << "Total events: " << nEntries_data << endl;
    cout << "Selected events: " << events_selected << endl;

    TCanvas *c = new TCanvas("c", "", 800, 600);
    c->SetLeftMargin(0.12);
    c->SetBottomMargin(0.12);

    h->SetLineColor(kBlue);
    h->SetFillColorAlpha(kBlue, 0.5);
    h->Draw("Hist");
    //c->SetLogy(true);
    c->SaveAs("fit_data.png");


    f_data->Close();

}
