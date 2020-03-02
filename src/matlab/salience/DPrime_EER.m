function [Dprime, EER]=DPrime_EER(ROC_struct)
FA=ROC_struct.FA;
Hits=ROC_struct.Hits;
FR=1-Hits;
[~, ind]=min(abs(FA-FR));
EER=FA(ind);
Dprime=sqrt(2)*norminv(ROC_struct.Area);
end