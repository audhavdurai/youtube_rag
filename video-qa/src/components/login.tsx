import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface UsernameModalProps {
  onSubmit: (username: string) => void;
}

const UsernameModal: React.FC<UsernameModalProps> = ({ onSubmit }) => {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) return;
    
    setLoading(true);
    await onSubmit(username.trim());
    setLoading(false);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-[#1a1d27] rounded-lg p-6 w-96">
        <h2 className="text-xl font-semibold text-gray-200 mb-6">
          Enter Your Username
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-[#090b10] text-gray-200 px-4 py-2 rounded-lg border border-gray-800 focus:outline-none focus:border-blue-500"
              placeholder="Username"
              required
              autoFocus
            />
          </div>

          <button
            type="submit"
            disabled={loading || !username.trim()}
            className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:bg-blue-800 disabled:text-gray-300 flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={18} />
                Loading...
              </>
            ) : (
              'Continue'
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

export default UsernameModal;