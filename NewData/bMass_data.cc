#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;
namespace fs = std::filesystem;


void bMass_data(){

    // Create output directory if it doesn't exist
    fs::create_directories("NewData/bMassPlots");

    // --- Open Files and Get Trees ---
    TFile *f = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple_flat_22F.root", "read");
	TTree *t = (TTree*)f->Get("ntuple");

    // --- Declare variables ---
    double bMass;
    double bVtxCL;
    double bCosAlphaBS;
    double bLBS;
    double tagB0;
    double bBarMass;

    t->SetBranchAddress("bMass", &bMass);
    t->SetBranchAddress("bVtxCL", &bVtxCL);
    t->SetBranchAddress("bCosAlphaBS", &bCosAlphaBS);
    t->SetBranchAddress("bLBS", &bLBS);
    t->SetBranchAddress("tagB0", &tagB0);
    t->SetBranchAddress("bBarMass", &bBarMass);

    double mmin = 5.0;
    double mmax = 5.6;
    double s_left = 5.15;
    double s_right = 5.4;
    
    // --- Create Histogram ---
    TH1D *h = new TH1D("h", "", 100, mmin, mmax);
    h->GetXaxis()->SetTitle("m(B^{0}) [GeV]");
    h->GetYaxis()->SetTitle("Events");

    Long64_t nEntries = t->GetEntries();
    int selected = 0;

    for (Long64_t i = 0; i < nEntries; ++i) {
        t->GetEntry(i);

        double mass_b = (tagB0 == 1) ? bMass : bBarMass;
		if (mass_b < mmin || mass_b > mmax) continue;

        // Fill histograms before cut
        if (bVtxCL < 0.2) continue;
        if (bCosAlphaBS < 0.9955) continue;
        if (bLBS < 0.05) continue;
        h->Fill(mass_b);
        selected++;
    }
    cout  << " Events selected: "<< selected << " / " << nEntries << endl;

    // --- Draw and save --- 
    TCanvas *c = new TCanvas("c", "");
    h->SetLineColor(kBlue);
    h->SetFillColorAlpha(kBlue, 0.5);
    h->Draw("HIST");
    c->SaveAs("NewData/bMassPlots/cuts3.pdf"); 
    delete c;
    delete h;
    f->Close();

}
