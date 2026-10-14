import assert from 'node:assert/strict';
import test from 'node:test';
test('initial product screen asks for clinical input without inventing patient results', async () => {
  const {default: worker}=await import('../dist/server/index.js');
  const response=await worker.fetch(new Request('http://localhost/',{headers:{accept:'text/html'}}),{ASSETS:{fetch:async()=>new Response('Not found',{status:404})}},{waitUntil(){},passThroughOnException(){}});
  assert.equal(response.status,200);
  const html=await response.text();
  assert.match(html,/Your clinical details/);
  assert.match(html,/For education and decision support only/);
  assert.doesNotMatch(html,/Your Vascularis insights/);
  assert.doesNotMatch(html,/TRACK A|stepNum|TabPFN|CatBoost/);
});
