import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import ReasoningPanel from '../../../components/arena/ReasoningPanel';

describe('ReasoningPanel', () => {
  it('returns null when reasoning is empty', () => {
    const { container } = render(<ReasoningPanel reasoning="" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders reasoning + token counts when present', () => {
    render(
      <ReasoningPanel reasoning="thinking..." inputTokens={42} outputTokens={7} />
    );
    expect(screen.getByText('thinking...')).toBeInTheDocument();
    expect(screen.getByText(/42 in \/ 7 out/)).toBeInTheDocument();
  });

  it('uses error styling when reasoning starts with the error glyph', () => {
    const { container } = render(
      <ReasoningPanel reasoning="⚠️ [SYSTEM ERROR] something" />
    );
    expect(container.querySelector('.bg-red-50')).not.toBeNull();
  });
});
