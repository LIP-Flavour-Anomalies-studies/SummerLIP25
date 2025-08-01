#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

void Plot_bMass(){

    // Create output directory if it doesn't exist
    fs::create_directories("Machine_Learning/Data_Application/Plots_bMass");
    fs::create_directories("Machine_Learning/Data_Application/Plots_bMass/Data");
    fs::create_directories("Machine_Learning/Data_Application/Plots_bMass/MC");


    // --- Open Files and Get Trees ---
    TFile *f = new TFile("Machine_Learning/Data_Application/ROOT/data_selected_mlFoM_output.root", "read");
	TTree *t = (TTree*)f->Get("Tdata");

    // --- Declare variables ---
    Double_t bTMass;
    Float_t score, threshold;

    t->SetBranchAddress("bTMass", &bTMass);
    
    // versions to loop over
    const int nVersions = 6;

    for (int v = 0; v < nVersions; ++v) {
        // --- Build branch names ---
        string score_name = Form("B_score_v%d", v);
        string thr_name   = Form("B_thr_v%d", v);

        // Connect model branches
        t->SetBranchAddress(score_name.c_str(), &score);
        t->SetBranchAddress(thr_name.c_str(), &threshold);

        // --- Create histogram for this version ---
        TH1D *h = new TH1D(Form("h_v%d", v), Form("Model v%d; m(B^{0}) [GeV/c^{2}]; Events", v), 100, 5.0, 5.6);
        h->SetLineColor(kBlue);
        h->SetFillColorAlpha(kBlue, 0.5);
        // h->SetDirectory(0); 

        Long64_t nEntries = t->GetEntries();
        int selected = 0;

        for (Long64_t i = 0; i < nEntries; ++i) {
            t->GetEntry(i);

            if (score > threshold) {
                h->Fill(bTMass);
                selected++;
            }
        }
        cout << "[v" << v << "] Events selected: " << selected << " / " << nEntries << endl;

        // --- Draw and save ---
        TCanvas *c = new TCanvas(Form("c_v%d", v), "", 800, 600);
        c->SetLeftMargin(0.12);
        c->SetBottomMargin(0.12);

        h->Draw("HIST");
        c->SaveAs(Form("Machine_Learning/Data_Application/Plots_bMass/Data/bMass_v%d.png", v));

        delete c;
        delete h;

    }

    f->Close();
    
}

