#include<iostream>
#include"ml/perceptron.hpp"
int main(void){
	{//AND
		ml::Perceptron p(2);
		p.set_weights({10,10,/*bias*/-15});
		std::cout<<"AND Gate:"<<std::endl;
		std::cout<<p.run({0,0})<<std::endl;//0	<-	3.05902e-07
		std::cout<<p.run({0,1})<<std::endl;//0	<-	0.00669285
		std::cout<<p.run({1,0})<<std::endl;//0	<-	0.00669285
		std::cout<<p.run({1,1})<<std::endl;//1	<-	0.993307
	}
	{//OR
		ml::Perceptron p(2);
		p.set_weights({15,15,/*bias*/-10});
		std::cout<<"OR Gate:"<<std::endl;
		std::cout<<p.run({0,0})<<std::endl;//0	<-	4.53979e-05
		std::cout<<p.run({0,1})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,0})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,1})<<std::endl;//1	<-	1
	}
	{//NAND
		ml::Perceptron p(2);
		p.set_weights({-10,-10,/*bias*/15});
		std::cout<<"NAND Gate:"<<std::endl;
		std::cout<<p.run({0,0})<<std::endl;//1	<-	1
		std::cout<<p.run({0,1})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,0})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,1})<<std::endl;//0	<-	0.00669285
	}
	{//XOR - composite
		std::vector<double>vor{15,15,/*bias*/-10};
		std::vector<double>vand{10,10,/*bias*/-15};
		std::vector<double>vnand{-10,-10,/*bias*/15};
		ml::Perceptron por(2);
		ml::Perceptron pand(2);
		ml::Perceptron pnand(2);
		por.set_weights(vor);
		pand.set_weights(vand);
		pnand.set_weights(vnand);
		auto run=[&pnand,&por,&pand](const double&a,const double&b){
			return pand.run({pnand.run({a,b}),por.run({a,b})});
		};
		std::cout<<"XOR Gate:"<<std::endl;
		std::cout<<run(0,0)<<std::endl;//0	<-	.00669585
		std::cout<<run(0,1)<<std::endl;//1	<-	0.992356
		std::cout<<run(1,0)<<std::endl;//1	<-	0.992356
		std::cout<<run(1,1)<<std::endl;//0	<-	0.00715281
	}
}
