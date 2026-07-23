import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './store/authStore';
import api from './api/client';
import { Layout } from './components/Layout';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { DealDetail } from './pages/DealDetail';
import { Recommendations } from './pages/Recommendations';
import { GlobalSHAP } from './pages/GlobalSHAP';
import { AdminRetrain } from './pages/AdminRetrain';
import { Loader2 } from 'lucide-react';

const queryClient = new QueryClient();

// A component to protect routes
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = useAuthStore(state => state.token);
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

// A component to handle demo data seeding on first load
function DemoSeeder({ children }: { children: React.ReactNode }) {
  const [seeding, setSeeding] = useState(false);
  const [seeded, setSeeded] = useState(localStorage.getItem('demo_seeded') === 'true');

  useEffect(() => {
    if (seeded) return;
    
    async function seedDemoData() {
      setSeeding(true);
      try {
        // Run simulator for 15 deals
        for (let i = 0; i < 15; i++) {
          await api.post('/crm/webhook/simulate', {
            platform: 'salesforce',
            opportunity_id: `OPP_${10000 + i}`,
            sales_agent: 'Sarah Chen',
            product: 'Enterprise Suite',
            engage_date: '2024-01-15',
            deal_stage: 'proposal'
          });
        }
        localStorage.setItem('demo_seeded', 'true');
        setSeeded(true);
      } catch (err) {
        console.error("Failed to seed demo data", err);
        // Even if it fails (e.g. timeout), let user proceed
        setSeeded(true);
      } finally {
        setSeeding(false);
      }
    }
    
    seedDemoData();
  }, [seeded]);

  if (seeding) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
        <Loader2 className="w-12 h-12 animate-spin text-brand-600 mb-4" />
        <h2 className="text-xl font-bold text-gray-900">Setting up demo data...</h2>
        <p className="text-gray-500 mt-2">This may take a few seconds.</p>
      </div>
    );
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          
          <Route path="/" element={
            <ProtectedRoute>
              <DemoSeeder>
                <Layout />
              </DemoSeeder>
            </ProtectedRoute>
          }>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="opportunities/:id" element={<DealDetail />} />
            <Route path="recommendations" element={<Recommendations />} />
            <Route path="analytics/features" element={<GlobalSHAP />} />
            <Route path="admin/retrain" element={<AdminRetrain />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
