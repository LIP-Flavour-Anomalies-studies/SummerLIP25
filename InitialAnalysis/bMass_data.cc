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
    double kstMass, kstBarMass;

    t_data->SetBranchAddress("bMass", &data_bMass);
    t_data->SetBranchAddress("bBarMass", &data_bBarMass);
    t_data->SetBranchAddress("tagB0", &data_tagB0);
    t_data->SetBranchAddress("kstMass", &kstMass);
    t_data->SetBranchAddress("kstBarMass", &kstBarMass);

    // --- Variables to apply cuts based on ROC analysis ---

    bool skipcuts = false; // true if don't want to apply cuts

    map<string, double> vars_cuts;
    map<string, double> cuts;

    vector<string> variables = {
        "bVtxCL", "bPt", "kstPt", "mumuMass", "mumuPt", 
        "kstTrkmPt", "kstTrkmDCABS", "kstTrkpPt", "kstTrkpDCABS",
        "mumPt", "mupPt", "bCosAlphaBS", "bLBS", "bDCABS",
    };
    
    if (!skipcuts){

        for (const auto &name : variables){
            vars_cuts[name] = 0;
            t_data->SetBranchAddress(name.c_str(), &vars_cuts[name]);
        }
        
        cuts = {
            {"bVtxCL", 0.024}, {"bPt", 11.392}, {"kstPt", 1.324}, {"mumuMass", 2.982}, {"mumuPt", 8.313},
            {"kstTrkmPt", 0.673}, {"kstTrkpPt", 0.621}, {"kstTrkmDCABS", -0.586}, {"kstTrkpDCABS", -0.657},
            {"mumPt", 3.101}, {"mupPt", 3.241}, {"bCosAlphaBS", 0.996}, {"bLBS", 0.024}, {"bDCABS", -0.004}
        };

    }


    // --- Create histogram ---
    TH1D *h = new TH1D("h", "", 100, 4.8, 5.8);
    h->GetXaxis()->SetTitle("m(B^{0}) Data [GeV/c^{2}]");
    h->GetYaxis()->SetTitle("Events");


    Long64_t nEntries_data = t_data->GetEntries();
    int events_selected = 0;
    for(Long64_t i = 0; i < nEntries_data; i++){    
        t_data->GetEntry(i);

        //Selection cuts based on ROC analysis
        
        if (!skipcuts){
            double mass_kst = (data_tagB0 == 1) ? kstMass : kstBarMass;
            if (mass_kst < 0.798) continue;

            bool pass_cuts = true;
            for (const auto &name : variables){
                if (vars_cuts[name] < cuts[name]){
                    pass_cuts = false;
                    break;
                }
            }
            if (!pass_cuts) continue;
        }

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
