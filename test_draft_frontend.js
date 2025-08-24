// Simple test to check if the backend /draft endpoint works
async function testDraftEndpoint() {
  try {
    console.log('Testing /draft endpoint...');
    
    const response = await fetch('http://localhost:5001/draft', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: 'Write a test email',
        tone: 'professional',
        length: 'medium'
      })
    });
    
    console.log('Response status:', response.status);
    console.log('Response headers:', [...response.headers.entries()]);
    
    if (response.ok) {
      const data = await response.json();
      console.log('Response data:', data);
    } else {
      const errorText = await response.text();
      console.error('Error response:', errorText);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
}

// Run the test
testDraftEndpoint();
