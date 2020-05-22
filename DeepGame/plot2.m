figure
id=32
lb=readtable('C:\Users\shaod\Desktop\Final_year_Project\DeepGame\exp_results\32\competitive\lb\32datalog.csv');
ub=readtable('C:\Users\shaod\Desktop\Final_year_Project\DeepGame\exp_results\32\competitive\ub\32datalog.csv');
Blue=[0 0.2 0.4];
coralRed=[0.7 0 0.12];
x1 = table2array(lb(:,1));
y1 = table2array(lb(:,2));
x0=0;
y0=0;
width=750;
height=450;
set(gcf,'position',[x0,y0,width,height]);
set(gca,'XTick',0:3)
lb=line(x1,y1,'Color',coralRed,'LineWidth',2);
ax1 = gca; % current axes
ax1.XColor = coralRed;
ax1.YColor = coralRed;
ax1.YLim=[0 0.023];
ax1.XLim=[0 2];
xlabel(ax1,'Iterations of Alpha-Beta Pruning ');
ylabel(ax1,'Lower Bounds (L2 Norm)');
ax1.LineWidth=1.5;

ax1_pos = ax1.Position; % position of first axes
ax2 = axes('Position',ax1_pos,...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'Color','none');
x2 = table2array(ub(:,1));
y2 = table2array(ub(:,2));
ax2.XColor = Blue;
ax2.YColor = Blue;
ax2.YLim=[1 8.1];
ub=line(x2,y2,'Parent',ax2,'Color',Blue,'LineWidth',2);
ax2.LineWidth=1.5;
xl2=xlabel(ax2,'Iterations of MCTS');
ylabel(ax2,'Uower Bounds (L2 Norm)');
xl2.Position(2)=xl2.Position(2)-0.2;
legend([ub,lb],{'Convergence Trends of Upper Bound','Convergence Trends of Lower Bound'})