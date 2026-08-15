import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area,
  BarChart, Bar, Cell
} from 'recharts';
import { 
  Activity, Thermometer, Wind, Droplets, HeartPulse, User, Bed, Clock, 
  Lock, ArrowRight, UploadCloud, FileText, CheckCircle, AlertTriangle, FilePlus
} from 'lucide-react';
import './index.css';
// Default to local Flask API when VITE_API_URL is not defined
const API = '/api'; // Use Vite proxy for backend calls

const getRiskLevel = (probability) => {
  if (probability < 0.30) return 'LOW';
  if (probability < 0.70) return 'MODERATE';
  return 'HIGH';
};

function App() {
  const [view, setView] = useState('login'); // login | selection | input | dashboard | train
  const [patientType, setPatientType] = useState('existing'); // existing | new | train
  const [authUser, setAuthUser] = useState(null); // {username, role}
  
  // Input State
  const [patientIdInput, setPatientIdInput] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  
  // Dashboard State
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // New Patient Input Method State
  const [inputMethod, setInputMethod] = useState('upload'); // upload | manual
  const [manualParams, setManualParams] = useState({
    HR: '', O2Sat: '', Temp: '', SBP: '', MAP: '', DBP: '', Resp: '',
    Age: '', ICULOS: '1', Creatinine: '', Platelets: '', Glucose: ''
  });

  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setManualParams(prev => ({ ...prev, [name]: value }));
  };

  const generatePsvFromForm = () => {
    const header = "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel";
    // Fill with values or 0/NaN
    const row = [
      manualParams.HR || '0', manualParams.O2Sat || '0', manualParams.Temp || '37', 
      manualParams.SBP || '0', manualParams.MAP || '0', manualParams.DBP || '0', 
      manualParams.Resp || '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
      manualParams.Creatinine || '0', '0', manualParams.Glucose || '0', '0', '0', '0', '0', '0',
      '0', '0', '0', '0', '0', manualParams.Platelets || '0', 
      manualParams.Age || '60', '0', '0', '0', '0', manualParams.ICULOS || '1', '0'
    ].join('|');
    return `${header}\n${row}`;
  };
  
  // Dashboard & UX State
  const [activeTab, setActiveTab] = useState('vitals'); // vitals | analysis | history
  
  // Note State
  const [doctorNote, setDoctorNote] = useState('');
  const [savedNotes, setSavedNotes] = useState([]);

  // === SHAP WATERFALL HELPER ===
  const processWaterfallData = (baseVal, features) => {
    if (!features || !features.length) return [];
    const reversedFeatures = [...features].reverse(); 
    let current = baseVal;
    
    const bottomUpData = reversedFeatures.map((f) => {
      const start = current;
      const end = current + f.value;
      current = end;
      return {
        name: f.feature,
        value: f.value,
        range: [start, end],
        isPositive: f.value > 0
      };
    });
    return bottomUpData.reverse(); // Largest effect at the top
  };

  // === API CALLS ===
  const fetchExistingPatient = async (id) => {
    setLoading(true); setError(null);
    try {
      const response = await fetch(`${API}/patient/${id}`).catch(() => {
        throw new Error("Cannot connect to backend. Is the Flask API running?");
      });
      let data;
      try { data = await response.json(); } catch (e) { throw new Error("Backend offline. Please run 'python src/api.py' first!"); }
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch patient data');
      }
      setPatientData(data);
      setView('dashboard');
    } catch (err) { setError(err.message); } 
    finally { setLoading(false); }
  };

  const predictRawPatient = async (psvData) => {
    setLoading(true); setError(null);
    try {
      const response = await fetch(`${API}/predict_raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: psvData }),
      }).catch(() => {
        throw new Error("Cannot connect to backend. Is the Flask API running?");
      });
      let data;
      try { data = await response.json(); } catch (e) { throw new Error("Backend offline. Please run 'python src/api.py' first!"); }
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process manual entry');
      }
      setPatientData(data);
      setView('dashboard');
    } catch (err) { setError(err.message); } 
    finally { setLoading(false); }
  };

  const uploadNewPatient = async (file) => {
    setLoading(true); setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${API}/predict_upload`, {
        method: 'POST',
        body: formData,
      }).catch(() => {
        throw new Error("Cannot connect to backend. Is the Flask API running?");
      });
      let data;
      try { data = await response.json(); } catch (e) { throw new Error("Backend offline. Please run 'python src/api.py' first!"); }
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process file');
      }
      setPatientData(data);
      setView('dashboard');
    } catch (err) { setError(err.message); } 
    finally { setLoading(false); }
  };

  // === PAGE 1: LOGIN ===
  const LoginPage = () => {
    const [loginErr, setLoginErr] = useState('');
    const handleLoginSubmit = async (e) => {
      e.preventDefault();
      setLoginErr('');
      const username = e.target.username.value;
      const password = e.target.password.value;
      try {
        const res = await fetch(`${API}/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        }).catch(() => { throw new Error("Backend offline. Please check if Flask server is running."); });
        
        let data = {};
        if (res.headers.get("content-type")?.includes("application/json")) {
          try {
            data = await res.json();
          } catch (_) {}
        }
        
        if (!res.ok) {
          throw new Error(data.error || `Login failed (Status ${res.status})`);
        }
        
        setAuthUser({ username: data.username, role: data.role });
        setView('selection');
      } catch (err) {
        setLoginErr(err.message);
      }
    };
    return (
      <div className="center-container">
        <div className="panel login-card">
          <div className="login-header">
            <Activity className="med-icon" />
            <h2>AI SEPSIS DETECTION</h2>
            <p style={{color: 'var(--text-muted)'}}>Hospital Staff Login</p>
          </div>
          <form className="login-form" onSubmit={handleLoginSubmit}>
            {loginErr && <div style={{color: 'red', marginBottom: '1rem', textAlign: 'center', fontSize: '0.9rem'}}>{loginErr}</div>}
            <div>
              <label style={{display:'block', marginBottom:'0.5rem', fontWeight:500}}>Username</label>
              <input name="username" type="text" placeholder="Staff ID" required />
            </div>
            <div>
              <label style={{display:'block', marginBottom:'0.5rem', fontWeight:500}}>Password</label>
              <input name="password" type="password" placeholder="••••••••" required />
            </div>
            <button type="submit" className="btn btn-primary" style={{marginTop:'1rem', width:'100%'}}>
              Login <ArrowRight size={18} />
            </button>
          </form>
          <div style={{marginTop: '1.5rem', padding: '0.85rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid var(--border-color)', textAlign: 'left', fontSize: '0.8rem'}}>
            <p style={{fontWeight: 600, color: 'var(--primary-color)', marginBottom: '0.4rem'}}>Demo Staff Credentials:</p>
            <ul style={{listStyleType: 'none', paddingLeft: 0, color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '4px'}}>
              <li>🏥 <strong>doctor1</strong> / docpass123 (Doctor role)</li>
              <li>🩺 <strong>nurse1</strong> / nursepass123 (Nurse role)</li>
              <li>🛡️ <strong>admin1</strong> / adminpass123 (Admin role)</li>
            </ul>
          </div>
        </div>
      </div>
    );
  };

  // === PAGE 2: SELECTION ===
  const SelectionPage = () => (
    <div className="center-container" style={{flexDirection: 'column'}}>
      <h2 className="selection-title">AI-Based Early Sepsis Detection</h2>
      {authUser && <p style={{marginBottom: '2rem', color: 'var(--text-muted)'}}>Logged in as: <strong>{authUser.username}</strong> ({authUser.role})</p>}
      <div className="selection-container">
        
        <div className="cards-grid">
          <div 
            className={`select-card panel ${patientType === 'existing' ? 'active' : ''}`}
            onClick={() => setPatientType('existing')}
          >
            <User className="card-icon" />
            <h3 style={{color: 'var(--primary-color)'}}>Existing Patient</h3>
            <p style={{color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem'}}>Fetch records from ICU database</p>
          </div>
          
          <div 
            className={`select-card panel ${patientType === 'new' ? 'active' : ''}`}
            onClick={() => setPatientType('new')}
          >
            <FilePlus className="card-icon" />
            <h3 style={{color: 'var(--primary-color)'}}>New Patient</h3>
            <p style={{color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem'}}>Upload new patient ICU data</p>
          </div>
        </div>

        <button className="btn btn-primary" onClick={() => {
          if (patientType === 'train') setView('train');
          else setView('input');
        }} style={{padding: '0.8rem 3rem'}}>
          Continue
        </button>
        <div style={{marginTop: '2rem'}}>
          <button className="btn btn-secondary" style={{border:'none'}} onClick={() => { setAuthUser(null); setView('login'); }}>&larr; Log Out</button>
        </div>
      </div>
    </div>
  );

  // === PAGE 3: INPUT ===
  const InputPage = () => {
    const handleSubmit = (e) => {
      e.preventDefault();
      if (patientType === 'existing' && patientIdInput) fetchExistingPatient(patientIdInput);
      else if (patientType === 'new') {
        if (inputMethod === 'upload' && selectedFile) uploadNewPatient(selectedFile);
        else if (inputMethod === 'manual') predictRawPatient(generatePsvFromForm());
      }
    };

    return (
      <div className="center-container" style={{flexDirection: 'column'}}>
        <h2 className="selection-title">Patient Data Retrieval</h2>
        <div className="panel input-container">
          <form onSubmit={handleSubmit}>
            {patientType === 'existing' ? (
              <div className="input-group">
                <label className="input-label">Enter Patient ID</label>
                <input 
                  type="text" 
                  autoFocus
                  style={{width:'100%', padding:'0.8rem', border:'1px solid var(--border-color)', borderRadius:'4px', fontSize:'1rem'}}
                  placeholder="e.g. p001000"
                  value={patientIdInput}
                  onChange={(e) => setPatientIdInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSubmit(e); }}
                  required
                />
              </div>
            ) : (
              <>
                <div className="input-method-toggle" style={{display:'flex', gap:'1rem', marginBottom:'1.5rem', justifyContent:'center'}}>
                  <button 
                    type="button"
                    className={`btn ${inputMethod === 'upload' ? 'btn-primary' : 'btn-secondary'}`} 
                    onClick={() => setInputMethod('upload')}
                    style={{flex:1, padding:'0.5rem'}}
                  >
                    Upload File
                  </button>
                  <button 
                    type="button"
                    className={`btn ${inputMethod === 'manual' ? 'btn-primary' : 'btn-secondary'}`} 
                    onClick={() => setInputMethod('manual')}
                    style={{flex:1, padding:'0.5rem'}}
                  >
                    Manual Entry
                  </button>
                </div>

                {inputMethod === 'upload' ? (
                  <div className="input-group">
                    <label className="input-label">Upload ICU Data (.csv or .xlsx)</label>
                    <div 
                      className="upload-area" 
                      onClick={() => document.getElementById('file-upload').click()}
                    >
                      <UploadCloud size={48} color="var(--secondary-color)" style={{margin:'0 auto 1rem'}} />
                      <p style={{color:'var(--primary-color)', fontWeight:500}}>
                        {selectedFile ? selectedFile.name : 'Click to browse files'}
                      </p>
                      <input 
                        id="file-upload" 
                        type="file" 
                        accept=".csv,.xlsx,.xls,.psv" 
                        style={{display:'none'}}
                        onChange={(e) => setSelectedFile(e.target.files[0])}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="input-group">
                    <label className="input-label" style={{textAlign:'center', fontSize:'1.1rem', marginBottom:'1.5rem'}}>Enter Patient Parameters</label>
                    <div className="params-form-grid" style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1rem'}}>
                      {[
                        { label: 'Heart Rate (bpm)', name: 'HR', placeholder: 'e.g. 85' },
                        { label: 'Temperature (°C)', name: 'Temp', placeholder: 'e.g. 37.5' },
                        { label: 'Resp. Rate (/min)', name: 'Resp', placeholder: 'e.g. 18' },
                        { label: 'MAP (mmHg)', name: 'MAP', placeholder: 'e.g. 80' },
                        { label: 'O2 Saturation (%)', name: 'O2Sat', placeholder: 'e.g. 98' },
                        { label: 'Creatinine (mg/dL)', name: 'Creatinine', placeholder: 'e.g. 1.1' },
                        { label: 'Platelets (count)', name: 'Platelets', placeholder: 'e.g. 250' },
                        { label: 'Glucose (mg/dL)', name: 'Glucose', placeholder: 'e.g. 110' },
                        { label: 'Age (years)', name: 'Age', placeholder: 'e.g. 65' },
                        { label: 'ICU Hours', name: 'ICULOS', placeholder: 'e.g. 6' },
                      ].map((field) => (
                        <div key={field.name} style={{textAlign:'left'}}>
                          <label style={{fontSize:'0.8rem', color:'var(--text-muted)', display:'block', marginBottom:'4px'}}>{field.label}</label>
                          <input 
                            type="number"
                            name={field.name}
                            value={manualParams[field.name]}
                            onChange={handleParamChange}
                            placeholder={field.placeholder}
                            style={{width:'100%', padding:'0.6rem', border:'1px solid var(--border-color)', borderRadius:'6px'}}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
            
            {error && <p style={{color: 'var(--accent-red)', marginBottom: '1rem', textAlign:'left'}}>{error}</p>}

            <button type="submit" className="btn btn-primary" style={{width:'100%', marginTop:'1rem'}} disabled={loading}>
              {loading ? 'Processing...' : (patientType === 'existing' ? 'Load Patient Data' : 'Predict Sepsis')}
            </button>
          </form>
        </div>
        <div style={{marginTop: '2rem'}}>
          <button className="btn btn-secondary" style={{border:'none', background:'transparent'}} onClick={() => setView('selection')}>
            &larr; Back to Selection
          </button>
        </div>
      </div>
    );
  };

  // === PAGE 4: DASHBOARD ===
  const DashboardPage = () => {
    if (!patientData) return null;

    const {
      patient_id, prediction, probability, severity, 
      vitals, risk_timeline, hr_timeline, shap, icu_hours, recommendations
    } = patientData;

    const probPercentage = (probability * 100).toFixed(1);
    const riskLevel = getRiskLevel(probability);
    const hrChartData = hr_timeline.map((h) => ({ time: h.time, value: h.value || 0 }));
    const riskChartData = risk_timeline.map((r) => ({ time: r.time, prob: r.prob || 0 }));

    const handleSaveNote = () => {
      if (!doctorNote.trim()) return;
      setSavedNotes([{ text: doctorNote, time: new Date().toLocaleTimeString() }, ...savedNotes]);
      setDoctorNote('');
    };

    return (
      <div className="dashboard">
        <div className="dashboard-scanline"></div>
        <header className="dashboard-header">
          <div style={{display:'flex', alignItems:'center', gap:'1rem'}}>
            <Activity color="var(--primary-color)" size={28} />
            <h1>AI-BASED EARLY SEPSIS DETECTION</h1>
          </div>
          <div style={{display:'flex', gap:'0.5rem'}}>
            <button className="btn btn-primary" onClick={() => window.print()} style={{background: 'var(--secondary-color)', border: 'none', display: 'flex', alignItems: 'center', gap: '8px'}}>
              <FileText size={18} /> Generate Report
            </button>
            <button className="btn btn-secondary" onClick={() => { 
              setView('selection'); setPatientIdInput(''); setSelectedFile(null); 
              setSavedNotes([]); setDoctorNote(''); setActiveTab('vitals');
            }}>
              Exit Dashboard
            </button>
          </div>
        </header>
        
        {/* Hidden Printable Report Component */}
        <div className="printable-report">
          <div className="report-header">
            <div style={{display:'flex', alignItems:'center', gap:'15px'}}>
              <Activity size={40} color="var(--primary-color)" />
              <div>
                <h1 style={{fontSize:'24px', margin:0, color:'var(--primary-color)'}}>AI-BASED EARLY SEPSIS DETECTION</h1>
                <p style={{margin:0, color:'var(--text-muted)', fontSize:'12px'}}>AI-Based Patient Risk Assessment Report</p>
              </div>
            </div>
            <div style={{textAlign:'right'}}>
              <p style={{margin:0}}><strong>Date:</strong> {new Date().toLocaleDateString()}</p>
              <p style={{margin:0}}><strong>Time:</strong> {new Date().toLocaleTimeString()}</p>
            </div>
          </div>

          <div className="report-section">
            <h2 className="report-section-title">Patient Identification</h2>
            <div className="report-grid">
              <div className="report-item"><span>Patient ID:</span> <strong>{patient_id}</strong></div>
              <div className="report-item"><span>Admission Time:</span> <strong>{icu_hours} Hours ICU</strong></div>
            </div>
          </div>

          <div className="report-section">
            <h2 className="report-section-title">Vital Signs Analysis</h2>
            <table className="report-table">
              <thead>
                <tr>
                  <th>Vital Sign</th>
                  <th>Current Value</th>
                  <th>Status</th>
                  <th>Clinical Indicator</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Heart Rate</td>
                  <td>{vitals.hr} bpm</td>
                  <td className={vitals.hr > 100 || vitals.hr < 60 ? 'text-danger' : ''}>{vitals.hr > 100 ? 'Tachycardia' : vitals.hr < 60 ? 'Bradycardia' : 'Normal'}</td>
                  <td>{Math.min(100, (vitals.hr / 1.2).toFixed(1))}% Baseline</td>
                </tr>
                <tr>
                  <td>Temperature</td>
                  <td>{vitals.temp} °C</td>
                  <td className={vitals.temp > 38 || vitals.temp < 36 ? 'text-danger' : ''}>{vitals.temp > 38 ? 'Fever' : vitals.temp < 36 ? 'Hypothermia' : 'Stable'}</td>
                  <td></td>
                </tr>
                <tr>
                  <td>Resp. Rate</td>
                  <td>{vitals.resp} /min</td>
                  <td className={vitals.resp > 20 ? 'text-danger' : ''}>{vitals.resp > 20 ? 'Tachypnea' : 'Normal'}</td>
                  <td>{Math.min(100, (vitals.resp * 4.5).toFixed(1))}% Normal</td>
                </tr>
                <tr>
                  <td>MAP</td>
                  <td>{vitals.map} mmHg</td>
                  <td className={vitals.map < 65 ? 'text-danger' : ''}>{vitals.map < 65 ? 'Hypotension' : 'Stable'}</td>
                  <td>Standard Perfusion</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="report-section" style={{background: riskLevel === 'HIGH' ? '#fff5f5' : riskLevel === 'MODERATE' ? '#fffbeb' : '#f0f9ff', padding: '20px', borderRadius: '8px', border: `2px solid ${riskLevel === 'HIGH' ? '#fecaca' : riskLevel === 'MODERATE' ? '#fef3c7' : '#bae6fd'}`}}>
            <h2 className="report-section-title" style={{borderBottomColor: riskLevel === 'HIGH' ? '#fecaca' : riskLevel === 'MODERATE' ? '#fef3c7' : '#bae6fd'}}>AI Sepsis Risk Assessment</h2>
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
              <div>
                <p style={{fontSize: '28px', fontWeight: 800, margin: '0', color: riskLevel === 'HIGH' ? '#dc2626' : riskLevel === 'MODERATE' ? '#d97706' : '#0369a1'}}>
                  {riskLevel === 'HIGH' ? 'HIGH RISK' : riskLevel === 'MODERATE' ? 'MODERATE RISK' : 'LOW RISK'}
                </p>
                <p style={{fontSize: '16px', margin: '5px 0', color: 'var(--text-main)'}}>
                  Risk Probability: <strong>{probPercentage}%</strong>
                </p>
              </div>
              <div style={{textAlign: 'right'}}>
                <p style={{margin: 0, fontSize: '14px', color: 'var(--text-muted)'}}>Severity Score</p>
                <p style={{margin: 0, fontSize: '24px', fontWeight: 700, color: 'var(--text-main)'}}>{severity}</p>
              </div>
            </div>
          </div>

          <div className="report-section">
            <h2 className="report-section-title">Clinical Recommendations</h2>
            <ul style={{paddingLeft: '20px', margin: 0}}>
              {recommendations?.map((rec, i) => (
                <li key={i} style={{marginBottom: '8px', fontSize: '14px', color: '#334155'}}>{rec}</li>
              ))}
            </ul>
          </div>

          <div className="report-footer">
            <div style={{borderTop: '1px solid #cbd5e1', paddingTop: '10px', display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: '#64748b'}}>
              <p>AI-generated risk assessment for educational and demonstration purposes.</p>
              <p>This system provides an AI-based risk assessment for educational and demonstration purposes and is not a substitute for professional medical judgment.</p>
              <div style={{display: 'flex', justifyContent: 'flex-end', marginTop: '-15px'}}>
                <p>Reference: {patient_id}-{Date.now()}</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Tab Navigation Menu */}
        <div className="tab-menu">
          <button 
            className={`tab-btn ${activeTab === 'vitals' ? 'active' : ''}`} 
            onClick={() => setActiveTab('vitals')}
          >
            <HeartPulse size={18} /> Patient Vitals
          </button>
          <button 
            className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`} 
            onClick={() => setActiveTab('analysis')}
          >
            <Activity size={18} /> AI Analysis
          </button>
          <button 
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} 
            onClick={() => setActiveTab('history')}
          >
            <Clock size={18} /> History & Notes
          </button>
        </div>

        <div className="dash-grid">
          {/* TOP: Patient Information (Always Visible) */}
          <div className="panel">
            <h2 className="panel-title"><User className="panel-icon"/> Patient Information</h2>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Patient ID</span>
                <span className="info-value">{patient_id}</span>
              </div>
              <div className="info-item">
                <span className="info-label">ICU Hours</span>
                <span className="info-value">
                  <Clock size={16} style={{verticalAlign:'middle', marginRight:'4px', color:'var(--text-muted)'}}/> 
                  {icu_hours}
                </span>
              </div>
              <div className="info-item">
                 <span className="info-label">Sepsis Risk Status</span>
                 <span className={`info-value ${riskLevel === 'HIGH' ? 'text-danger' : riskLevel === 'MODERATE' ? 'text-warning' : 'text-safe'}`} style={{fontWeight: 700}}>
                   {riskLevel === 'HIGH' ? 'HIGH RISK' : riskLevel === 'MODERATE' ? 'MODERATE RISK' : 'LOW RISK'}
                 </span>
              </div>
            </div>
          </div>

          {/* TAB 1: Patient Vitals */}
          {activeTab === 'vitals' && (
            <div className="middle-grid">
              <div className="panel">
                <h2 className="panel-title"><HeartPulse className="panel-icon"/> Vital Signs Monitor</h2>
                <div className="vitals-grid">
                  <div className="vital-card">
                    <HeartPulse className="vital-icon pulse-animation" />
                    <div>
                      <div className="vital-name">Heart Rate</div>
                      <div className="vital-val alert-flash">
                        {vitals.hr} <span style={{fontSize:'1rem', color:'var(--text-muted)'}}>bpm</span>
                      </div>
                    </div>
                  </div>
                  <div className="vital-card">
                    <Thermometer className="vital-icon" />
                    <div>
                      <div className="vital-name">Temperature</div>
                      <div className="vital-val">{vitals.temp} <span style={{fontSize:'0.9rem', color:'var(--text-muted)'}}>&deg;C</span></div>
                    </div>
                  </div>
                  <div className="vital-card">
                    <Wind className="vital-icon" />
                    <div>
                      <div className="vital-name">Respiratory Rate</div>
                      <div className="vital-val">{vitals.resp} <span style={{fontSize:'0.9rem', color:'var(--text-muted)'}}>/min</span></div>
                    </div>
                  </div>
                  <div className="vital-card">
                    <Activity className="vital-icon pulse-animation" style={{animationDelay: '0.5s'}} />
                    <div>
                      <div className="vital-name">MAP</div>
                      <div className="vital-val">{vitals.map} <span style={{fontSize:'1rem', color:'var(--text-muted)'}}>mmHg</span></div>
                    </div>
                  </div>
                  <div className="vital-card">
                    <Droplets className="vital-icon" />
                    <div>
                      <div className="vital-name">Creatinine</div>
                      <div className="vital-val">{vitals.creatinine}</div>
                    </div>
                  </div>
                  <div className="vital-card">
                    <Activity className="vital-icon" />
                    <div>
                      <div className="vital-name">Platelets</div>
                      <div className="vital-val">{vitals.platelets}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="panel" style={{textAlign: 'center', display:'flex', flexDirection:'column', justifyContent:'center'}}>
                <h2 className="panel-title" style={{justifyContent:'center'}}><AlertTriangle className="panel-icon"/> Prediction Summary</h2>
                
                <div className={`pred-risk ${riskLevel === 'HIGH' ? 'text-danger' : riskLevel === 'MODERATE' ? 'text-warning' : 'text-safe'}`}>
                  {riskLevel === 'HIGH' ? 'HIGH RISK' : riskLevel === 'MODERATE' ? 'MODERATE RISK' : 'LOW RISK'}
                </div>
                
                <div className="pred-prob">
                  Risk Probability: <strong>{probPercentage}%</strong>
                </div>
                <div style={{color: 'var(--text-muted)'}}>
                  Severity Level: <strong className={riskLevel === 'HIGH' ? 'text-warning' : 'text-safe'}>{severity}</strong>
                </div>

                <div className="meter-wrapper">
                  <div style={{fontSize:'0.85rem', color:'var(--text-muted)', marginBottom:'0.5rem', textAlign:'left'}}>Risk Meter</div>
                  <div className="meter-bg">
                    <div 
                      className={`meter-fill ${riskLevel === 'HIGH' ? 'fill-danger' : riskLevel === 'MODERATE' ? 'fill-warn' : 'fill-safe'}`} 
                      style={{ width: `${probPercentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: AI Analysis & Graphs */}
          {activeTab === 'analysis' && (
            <>
              <div className="graphs-grid">
                <div className="panel">
                  <h2 className="panel-title">ICU Heart Rate Trend (HR vs Time)</h2>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={hrChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="time" stroke="#64748b" />
                        <YAxis stroke="#64748b" domain={['dataMin - 10', 'dataMax + 10']} />
                        <RechartsTooltip />
                        <Line type="monotone" dataKey="value" stroke="var(--primary-color)" strokeWidth={3} dot={{ fill: 'var(--primary-color)', r: 4 }} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="panel">
                  <h2 className="panel-title">Sepsis Risk Progression</h2>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={riskChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="time" stroke="#64748b" />
                        <YAxis stroke="#64748b" domain={[0, 1]} />
                        <RechartsTooltip />
                        <Area type="monotone" dataKey="prob" stroke="#ef4444" fillOpacity={1} fill="url(#colorProb)" strokeWidth={3} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="bottom-grid">
                <div className="panel" style={{height:'350px'}}>
                  <h2 className="panel-title"><FileText className="panel-icon"/> Explainable AI Panel (SHAP Waterfall)</h2>
                  <div style={{color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem', textAlign: 'center'}}>
                    Base Value (E[f(x)]): {patientData.shap_base_value?.toFixed(3)}
                  </div>
                  <div className="chart-container" style={{height: 'calc(100% - 3rem)'}}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart 
                        data={processWaterfallData(patientData.shap_base_value || 0, shap || [])} 
                        layout="vertical" 
                        margin={{ top: 10, right: 30, left: 60, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                        <XAxis type="number" hide={false} stroke="#64748b" />
                        <YAxis type="category" dataKey="name" width={130} tick={{fontSize: 11, fill: 'var(--text-main)'}} stroke="#64748b" />
                        <RechartsTooltip 
                          cursor={{fill: 'rgba(0,0,0,0.03)'}}
                          formatter={(value, name, props) => {
                             const v = props.payload.value;
                             return [v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3), 'Impact'];
                          }}
                        />
                        <Bar dataKey="range" isAnimationActive={false}>
                          {processWaterfallData(patientData.shap_base_value || 0, shap || []).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.isPositive ? '#ef4444' : '#38bdf8'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="panel">
                  <h2 className="panel-title"><CheckCircle className="panel-icon" style={{color:'var(--accent-green)'}}/> Clinical Decision Support</h2>
                  <ul className="cds-list">
                    {recommendations?.map((rec, i) => (
                      <li key={i} className="cds-item">
                         <AlertTriangle size={16} color="var(--primary-color)" style={{flexShrink:0, marginTop:'2px'}}/>
                         <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                  <div style={{marginTop: '2rem', padding: '1rem', background: '#e0f2fe', borderRadius: '8px', color: 'var(--primary-color)', fontSize: '0.9rem', textAlign:'center'}}>
                    <FileText size={16} style={{verticalAlign:'middle', marginRight:'6px'}}/>
                    AI-generated risk assessment based on the provided patient data.
                  </div>
                </div>
              </div>
            </>
          )}

          {/* TAB 3: History & Notes */}
          {activeTab === 'history' && (
            <div className="history-notes-grid">
              {/* Patient History Panel */}
              <div className="panel">
                <h2 className="panel-title"><Clock className="panel-icon"/> Patient History Panel</h2>
                <div className="table-responsive">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Probability</th>
                        <th>Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...risk_timeline].reverse().map((r, idx) => {
                        const probVal = parseFloat(r.prob);
                        const histRisk = getRiskLevel(probVal);
                        let resText = "LOW RISK";
                        let resClass = "text-safe";
                        if (histRisk === 'HIGH') {
                          resText = "HIGH RISK";
                          resClass = "text-danger";
                        } else if (histRisk === 'MODERATE') {
                          resText = "MODERATE RISK";
                          resClass = "text-warning";
                        }

                        return (
                          <tr key={idx}>
                            <td>{r.time}</td>
                            <td>{(probVal * 100).toFixed(1)}%</td>
                            <td className={resClass} style={{fontWeight: 600}}>{resText}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Doctor Notes Section */}
              <div className="panel" style={{display: 'flex', flexDirection: 'column'}}>
                <h2 className="panel-title"><FileText className="panel-icon"/> Doctor Notes</h2>
                <div className="notes-container" style={{flex: 1, overflowY: 'auto', marginBottom: '1rem'}}>
                  {savedNotes.length === 0 ? (
                    <p style={{color: 'var(--text-muted)', fontSize: '0.9rem', fontStyle: 'italic'}}>No notes recorded yet.</p>
                  ) : (
                    savedNotes.map((note, idx) => (
                      <div key={idx} className="note-item">
                        <div className="note-time">{note.time}</div>
                        <div className="note-text">{note.text}</div>
                      </div>
                    ))
                  )}
                </div>
                <div className="note-input-area">
                  <textarea 
                    className="note-textarea"
                    placeholder="Enter notes (Ctrl+Enter to save)"
                    value={doctorNote}
                    onChange={(e) => setDoctorNote(e.target.value)}
                    onKeyDown={(e) => {
                      // Allow Ctrl + Enter (or Cmd + Enter on Mac) to submit note automatically without touching mouse
                      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        handleSaveNote();
                      }
                    }}
                    autoFocus
                    rows={3}
                  ></textarea>
                  <button className="btn btn-primary" style={{width: '100%', marginTop: '0.5rem'}} onClick={handleSaveNote}>
                    Save Note
                  </button>
                </div>
              </div>
              
              {/* Feature Importance Summary */}
              <div className="panel">
                <h2 className="panel-title"><AlertTriangle className="panel-icon"/> Top Risk Factors</h2>
                <div className="risk-factors-list">
                  {shap && shap.length > 0 ? (
                    shap.slice(0, 3).map((f, idx) => {
                      const isIncrease = f.value > 0;
                      return (
                         <div key={idx} className="risk-factor-item">
                           <span className="risk-num">{idx + 1}️⃣</span>
                           <span className="risk-name">{f.feature}</span>
                           <span className={`risk-arrow ${isIncrease ? 'text-danger' : 'text-safe'}`}>
                             {isIncrease ? '↑' : '↓'}
                           </span>
                         </div>
                      );
                    })
                  ) : (
                    <p style={{color: 'var(--text-muted)', fontSize:'0.9rem'}}>No risk factor data available.</p>
                  )}
                </div>
                <p style={{fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem'}}>
                  Risk factors are computed using SHAP values based on current vital signs correlation to ICU sepsis onset.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // === PAGE X: TRAIN PAGE ===
  const TrainPage = () => {
    const [trainStatus, setTrainStatus] = useState('');
    const [trainLogs, setTrainLogs] = useState([]);
    
    const startTraining = async () => {
        setTrainStatus('Starting...');
        try {
            const res = await fetch(`${API}/train`, { method: 'POST' });
            const data = await res.json();
            if(!res.ok) throw new Error(data.message || 'Failed');
            setTrainStatus('Training in progress...');
            
            const interval = setInterval(async () => {
                const statRes = await fetch(`${API}/train_status`);
                const statData = await statRes.json();
                setTrainLogs(statData.logs || []);
                if(!statData.is_training) {
                    clearInterval(interval);
                    setTrainStatus(statData.result || 'Finished with errors: ' + statData.error);
                }
            }, 2000);
        } catch(e) {
            setTrainStatus('Error: ' + e.message);
        }
    };
    
    return (
      <div className="center-container" style={{flexDirection: 'column'}}>
        <div className="panel" style={{width: '600px', maxWidth: '90%'}}>
           <h2>Terminal - Train ML Model</h2>
           <p style={{color: 'var(--text-muted)', marginBottom: '1rem'}}>
               Only authorized personnel (Admin) can retrain the AI with newly obtained system data.
           </p>
           <button className="btn btn-primary" onClick={startTraining}>Start Pipeline</button>
           <div style={{marginTop: '1rem', padding: '1rem', background: '#f8fafc', height: '200px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px', fontFamily: 'monospace', fontSize: '13px'}}>
               <strong>{trainStatus}</strong><br/>
               {trainLogs.map((l, i) => <div key={i}>{l}</div>)}
           </div>
           <button className="btn btn-secondary" onClick={() => setView('selection')} style={{marginTop: '1.5rem'}}>Back to Dashboard</button>
        </div>
      </div>
    );
  };

  // === RENDERING ROUTER ===
  if (view === 'login') return <LoginPage />;
  if (view === 'selection') return <SelectionPage />;
  if (view === 'input') return <InputPage />;
  if (view === 'dashboard') return DashboardPage();
  if (view === 'train') return <TrainPage />;
  
  return null;
}

export default App;
