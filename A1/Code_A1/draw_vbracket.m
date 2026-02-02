function draw_vbracket(x, y1, y2, label)
% Draw a simple vertical bracket from y1 to y2 at x with a label.
    plot([x x], [y1 y2], 'k-', 'LineWidth', 1.5);
    ax = gca; xr = ax.XLim;
    tick = 0.01*(xr(2)-xr(1));
    plot([x-tick x+tick], [y1 y1], 'k-', 'LineWidth', 1.5);
    plot([x-tick x+tick], [y2 y2], 'k-', 'LineWidth', 1.5);
    text(x + 0.01*(xr(2)-xr(1)), (y1+y2)/2, label, 'FontWeight','bold');
end