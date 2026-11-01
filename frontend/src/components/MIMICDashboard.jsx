import React, { useState, lazy, Suspense } from 'react';
const HumanRiskModel3D = lazy(() => import('./HumanRiskModel3D'));

function probability(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function PredictionCard({ title, probability: score, prediction, color }) {
  const positive = Number(prediction) === 1;

  return (
    <div className="glass-panel p-5 rounded-xl space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
        <span className={`text-xs px-2 py-1 rounded-md ${
          positive
            ? 'bg-amber-500/10 text-amber-300'
            : 'bg-emerald-500/10 text-emerald-300'
        }`}>
          {positive ? 'Positive prediction' : 'Negative prediction'}
        </span>
      </div>

      <div className={`text-3xl font-bold ${color}`}>
        {probability(score)}
      </div>

      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: probability(score),
            backgroundColor: color === 'text-cyan-300' ? '#22d3ee' : '#a78bfa',
          }}
        />
      </div>

      <p className="text-xs text-slate-500">
        Model output at the selected assessment
      </p>
    </div>
  );
}

export default function MIMICDashboard({
  roster,
  selectedStay,
  onSelectStay,
  patient,
  trajectory,
  loading,
}) {
  const [selectedRiskModel, setSelectedRiskModel] = useState('bilstm');
  const latest = patient?.latest_assessment;
  const selectedRisk = selectedRiskModel === 'bilstm'
    ? latest?.bilstm_probability
    : latest?.transformer_probability;
  const assessments = trajectory?.trajectory || [];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-100">
            MIMIC-IV Retrospective Analysis
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Held-out test cohort · Independent BiLSTM and Transformer models
          </p>
        </div>

        <span className="px-3 py-1.5 rounded-lg text-xs border border-amber-700/40 bg-amber-950/30 text-amber-300">
          Research use only · Not live clinical data
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="glass-panel p-4 rounded-xl">
          <p className="text-xs text-slate-400">Patients in test cohort</p>
          <p className="text-2xl font-bold text-cyan-300 mt-1">
            {roster?.patient_count ?? '—'}
          </p>
        </div>
        <div className="glass-panel p-4 rounded-xl">
          <p className="text-xs text-slate-400">ICU stays in test cohort</p>
          <p className="text-2xl font-bold text-purple-300 mt-1">
            {roster?.icu_stay_count ?? '—'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <section className="glass-panel rounded-xl p-3 lg:col-span-1">
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            ICU stays
          </h3>

          <div className="space-y-2 max-h-[520px] overflow-y-auto">
            {(roster?.patients || []).map((item) => (
              <button
                key={item.stay_id}
                onClick={() => onSelectStay(item.stay_id)}
                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                  String(selectedStay) === String(item.stay_id)
                    ? 'bg-cyan-500/10 border-cyan-500/40'
                    : 'bg-slate-900/40 border-slate-800 hover:border-slate-600'
                }`}
              >
                <div className="text-xs font-semibold text-slate-200">
                  Stay {item.stay_id}
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  Patient {item.subject_id}
                </div>
                <div className="flex justify-between gap-2 mt-2 text-[10px]">
                  <span className="text-cyan-300">
                    BiLSTM {probability(item.bilstm_probability)}
                  </span>
                  <span className="text-purple-300">
                    Transformer {probability(item.transformer_probability)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="lg:col-span-3 space-y-4">
          {loading ? (
            <div className="glass-panel p-8 rounded-xl text-center text-sm text-slate-400">
              Loading MIMIC patient data...
            </div>
          ) : latest ? (
            <>
              <div className="glass-panel p-4 rounded-xl">
                <h3 className="text-sm font-semibold text-slate-200">
                  Selected ICU stay
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3 text-xs">
                  <div>
                    <p className="text-slate-500">Patient ID</p>
                    <p className="text-slate-200 mt-1">{latest.subject_id}</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Admission ID</p>
                    <p className="text-slate-200 mt-1">{latest.hadm_id}</p>
                  </div>
                  <div>
                    <p className="text-slate-500">ICU stay ID</p>
                    <p className="text-slate-200 mt-1">{latest.stay_id}</p>
                  </div>
                </div>
                <p className="text-xs text-slate-400 mt-3">
                  Latest assessment: {latest.assessment_time}
                </p>
              </div>

              <div className="glass-panel rounded-xl p-4 space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-200">
                    Interactive Human Risk Model
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Select an independent model to visualize its predicted risk.
                  </p>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedRiskModel('bilstm')}
                    className={`px-4 py-2 rounded-lg text-xs font-semibold transition-colors ${
                      selectedRiskModel === 'bilstm'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    BiLSTM
                  </button>

                  <button
                    onClick={() => setSelectedRiskModel('transformer')}
                    className={`px-4 py-2 rounded-lg text-xs font-semibold transition-colors ${
                      selectedRiskModel === 'transformer'
                        ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    Transformer
                  </button>
                </div>

                <Suspense fallback={
                  <div className="h-[380px] flex items-center justify-center text-sm text-slate-400">
                    Loading 3D visualization...
                  </div>
                }>
                  <HumanRiskModel3D
                    riskProbability={selectedRisk}
                    modelName={selectedRiskModel === 'bilstm' ? 'BiLSTM' : 'Transformer'}
                  />
                </Suspense>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <PredictionCard
                  title="BiLSTM"
                  probability={latest.bilstm_probability}
                  prediction={latest.bilstm_prediction}
                  color="text-cyan-300"
                />
                <PredictionCard
                  title="Transformer"
                  probability={latest.transformer_probability}
                  prediction={latest.transformer_prediction}
                  color="text-purple-300"
                />
              </div>

              <div className="glass-panel p-4 rounded-xl">
                <h3 className="text-sm font-semibold text-slate-200 mb-3">
                  Prediction trajectory
                </h3>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="text-slate-400 border-b border-slate-700">
                      <tr>
                        <th className="py-3 pr-3">Assessment time</th>
                        <th className="py-3 px-3">BiLSTM</th>
                        <th className="py-3 px-3">BiLSTM result</th>
                        <th className="py-3 px-3">Transformer</th>
                        <th className="py-3 px-3">Transformer result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {assessments.map((item, index) => (
                        <tr
                          key={`${item.assessment_time}-${index}`}
                          className="border-b border-slate-800/70 text-slate-300"
                        >
                          <td className="py-3 pr-3 whitespace-nowrap">
                            {item.assessment_time}
                          </td>
                          <td className="py-3 px-3">
                            {probability(item.bilstm_probability)}
                          </td>
                          <td className="py-3 px-3">
                            {Number(item.bilstm_prediction) === 1 ? 'Positive' : 'Negative'}
                          </td>
                          <td className="py-3 px-3">
                            {probability(item.transformer_probability)}
                          </td>
                          <td className="py-3 px-3">
                            {Number(item.transformer_prediction) === 1 ? 'Positive' : 'Negative'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {assessments.length === 0 && (
                  <p className="text-xs text-slate-500 py-4">
                    No trajectory assessments available.
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="glass-panel p-8 rounded-xl text-center text-sm text-slate-400">
              Select an ICU stay to view its predictions.
            </div>
          )}
        </section>
      </div>

      <p className="text-[10px] text-slate-500 leading-relaxed">
        MIMIC-IV retrospective held-out test data. Predictions are research
        outputs, not clinical diagnoses or treatment recommendations.
        BiLSTM and Transformer outputs are displayed independently.
      </p>
    </div>
  );
}


