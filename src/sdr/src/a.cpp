#include<iostream>
#include"ml/perceptron.hpp"
#include"ml/multilayerperceptron.hpp"
int main(void){
	{
		/*
		 * 7 to 1 seven segment display
		 *  5
		 * 4 0
		 *  6
		 * 3 1
		 *  2
		 */
		double MSE=0.0;
		constexpr int epochs=3000;
		auto sdrp=ml::MultilayerPerceptron({/*in*/7,/*layers*/7,/*outputs*/1});
		for(int i=0;i<epochs;i++){
			MSE=0.0;
			//            0 1 2 3 4 5 6
			MSE+=sdrp.bp({1,1,1,1,1,1,0},{0.05});//0
			MSE+=sdrp.bp({0,1,1,0,0,0,0},{0.15});//1
			MSE+=sdrp.bp({1,1,0,1,1,0,1},{0.25});//2
			MSE+=sdrp.bp({1,1,1,1,0,0,1},{0.35});//3
			MSE+=sdrp.bp({0,1,1,0,0,1,1},{0.45});//4
			MSE+=sdrp.bp({1,0,1,1,0,1,1},{0.55});//5
			MSE+=sdrp.bp({1,0,1,1,1,1,1},{0.65});//6
			MSE+=sdrp.bp({1,1,1,0,0,0,0},{0.75});//7
			MSE+=sdrp.bp({1,1,1,1,1,1,1},{0.85});//8
			MSE+=sdrp.bp({1,1,1,1,0,1,1},{0.95});//9
		}
		MSE/=10.0;
		std::cout<<"7 to 1:MSE:"<<MSE<<std::endl;
	}
	{
		/*
		 * 7 to 10 seven segment display
		 *  5
		 * 4 0
		 *  6
		 * 3 1
		 *  2
		 */
		double MSE=0.0;
		constexpr int epochs=3000;
		auto sdrp=ml::MultilayerPerceptron({/*in*/7,/*layers*/7,/*output*/10});
		for(int i=0;i<epochs;i++){
			MSE=0.0;
			//            0 1 2 3 4 5 6
			MSE+=sdrp.bp({1,1,1,1,1,1,0},{1,0,0,0,0,0,0,0,0,0});//0
			MSE+=sdrp.bp({0,1,1,0,0,0,0},{0,1,0,0,0,0,0,0,0,0});//1
			MSE+=sdrp.bp({1,1,0,1,1,0,1},{0,0,1,0,0,0,0,0,0,0});//2
			MSE+=sdrp.bp({1,1,1,1,0,0,1},{0,0,0,1,0,0,0,0,0,0});//3
			MSE+=sdrp.bp({0,1,1,0,0,1,1},{0,0,0,0,1,0,0,0,0,0});//4
			MSE+=sdrp.bp({1,0,1,1,0,1,1},{0,0,0,0,0,1,0,0,0,0});//5
			MSE+=sdrp.bp({1,0,1,1,1,1,1},{0,0,0,0,0,0,1,0,0,0});//6
			MSE+=sdrp.bp({1,1,1,0,0,0,0},{0,0,0,0,0,0,0,1,0,0});//7
			MSE+=sdrp.bp({1,1,1,1,1,1,1},{0,0,0,0,0,0,0,0,1,0});//8
			MSE+=sdrp.bp({1,1,1,1,0,1,1},{0,0,0,0,0,0,0,0,0,1});//9
		}
		MSE/=10.0;
		std::cout<<"7 to 10:MSE:"<<MSE<<std::endl;
	}
	{
		/*
		 * 7 to 7 seven segment display
		 *  5
		 * 4 0
		 *  6
		 * 3 1
		 *  2
		 */
		double MSE=0.0;
		constexpr int epochs=3000;
		auto sdrp=ml::MultilayerPerceptron({/*in*/7,/*layers*/7,/*output*/7});
		for(int i=0;i<epochs;i++){
			MSE=0.0;
			//            0 1 2 3 4 5 6
			MSE+=sdrp.bp({1,1,1,1,1,1,0},{1,1,1,1,1,1,0});//0
			MSE+=sdrp.bp({0,1,1,0,0,0,0},{0,1,1,0,0,0,0});//1
			MSE+=sdrp.bp({1,1,0,1,1,0,1},{1,1,0,1,1,0,1});//2
			MSE+=sdrp.bp({1,1,1,1,0,0,1},{1,1,1,1,0,0,1});//3
			MSE+=sdrp.bp({0,1,1,0,0,1,1},{0,1,1,0,0,1,1});//4
			MSE+=sdrp.bp({1,0,1,1,0,1,1},{1,0,1,1,0,1,1});//5
			MSE+=sdrp.bp({1,0,1,1,1,1,1},{1,0,1,1,1,1,1});//6
			MSE+=sdrp.bp({1,1,1,0,0,0,0},{1,1,1,0,0,0,0});//7
			MSE+=sdrp.bp({1,1,1,1,1,1,1},{1,1,1,1,1,1,1});//8
			MSE+=sdrp.bp({1,1,1,1,0,1,1},{1,1,1,1,0,1,1});//9
		}
		MSE/=10.0;
		std::cout<<"7 to 7:MSE:"<<MSE<<std::endl;
	}

}
