import numpy as np 
a=[1,2,3,4,5,6]
b=np.cumsum(a)
print(b)
c=np.array([True,False,True,False,False])
d=~c         #[False  True False  True  True]
e=sum(d)     # 3   True的个数
print(d)
print(e,"\n")
x=np.array([0,1,1,0.66,0.5,0.4,0.5,0.43,0.375,0.44,0.5])
for i in range(x.size-1,0,-1):
     print(i)
y=list(["1","5","abc"])
with open("test.txt","w") as f:
    f.writelines(y)
